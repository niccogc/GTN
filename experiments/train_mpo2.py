# type: ignore
"""
Production training script for MPO2 model with multi-seed analysis.
Configured via environment variables.
"""
import os
import sys
import json
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.NTN import NTN
from model.losses import MSELoss, CrossEntropyLoss
from model.utils import REGRESSION_METRICS, CLASSIFICATION_METRICS, compute_quality, create_inputs
from model.MPO2_models import MPO2
from experiments.dataset_loader import load_dataset
from experiments.trackers import create_tracker

torch.set_default_dtype(torch.float32)


def parse_env():
    """Parse configuration from environment variables."""
    config = {
        # Task type
        'task': os.getenv('TASK', 'regression').lower(),  # 'regression' or 'classification'
        
        # Dataset
        'dataset': os.getenv('DATASET', 'california_housing'),
       
        # Model architecture
        'L': int(os.getenv('L', '3')),
        'bond_dim': int(os.getenv('BOND_DIM', '6')),
        'output_site': int(os.getenv('OUTPUT_SITE', '1')),
        'init_strength': float(os.getenv('INIT_STRENGTH', '0.1')),
        
        # Training
        'batch_size': int(os.getenv('BATCH_SIZE', '100')),
        'n_epochs': int(os.getenv('N_EPOCHS', '10')),
        'jitter_start': float(os.getenv('JITTER_START', '0.05')),
        'jitter_decay': float(os.getenv('JITTER_DECAY', '0.9')),
        'jitter_min': float(os.getenv('JITTER_MIN', '1e-6')),
        
        # Multi-seed
        'seeds': [int(s) for s in os.getenv('SEEDS', '0,1,2,3,4').split(',')],
        
        # Output
        'output_file': os.getenv('OUTPUT_FILE', 'results_mpo2.json'),
        'verbose': os.getenv('VERBOSE', 'false').lower() == 'true',
        
        # Tracking
        'tracker_backend': os.getenv('TRACKER_BACKEND', 'file'),  # 'file', 'aim', 'both', 'none'
        'tracker_dir': os.getenv('TRACKER_DIR', 'experiment_logs'),
        'aim_repo': os.getenv('AIM_REPO', None),  # AIM repo (None = local .aim)
        'experiment_name': os.getenv('EXPERIMENT_NAME', 'mpo2_experiment')
    }
    
    # Validate task type
    if config['task'] not in ['regression', 'classification']:
        raise ValueError(f"TASK must be 'regression' or 'classification', got: {config['task']}")
    
    return config


def train_single_seed(config, seed, data, input_dim, output_dim, tracker=None):
    """Train model with a single seed."""
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Select loss function and metrics based on task
    if config['task'] == 'regression':
        loss_fn = MSELoss()
        eval_metrics = REGRESSION_METRICS
    else:  # classification
        loss_fn = CrossEntropyLoss()
        eval_metrics = CLASSIFICATION_METRICS
    
    # Create jitter schedule
    jitter_schedule = [
        max(config['jitter_start'] * (config['jitter_decay'] ** epoch), config['jitter_min'])
        for epoch in range(config['n_epochs'])
    ]
    
    # Create model
    model = MPO2(
        L=config['L'],
        bond_dim=config['bond_dim'],
        phys_dim=input_dim,
        output_dim=output_dim,
        output_site=config['output_site'],
        init_strength=config['init_strength']
    )
    
    # Create data loaders for train, val, test
    loader_train = create_inputs(
        X=data['X_train'], y=data['y_train'],
        input_labels=model.input_labels,
        output_labels=model.output_dims,
        batch_size=config['batch_size']
    )
    
    loader_val = create_inputs(
        X=data['X_val'], y=data['y_val'],
        input_labels=model.input_labels,
        output_labels=model.output_dims,
        batch_size=config['batch_size']
    )
    
    loader_test = create_inputs(
        X=data['X_test'], y=data['y_test'],
        input_labels=model.input_labels,
        output_labels=model.output_dims,
        batch_size=config['batch_size']
    )
    
    # Create NTN wrapper (with training data)
    ntn = NTN(
        tn=model.tn,
        output_dims=model.output_dims,
        input_dims=model.input_dims,
        loss=loss_fn,
        data_stream=loader_train
    )
    
    # Setup callbacks for tracking
    def callback_init(scores_train, scores_val, info):
        if tracker:
            # Log hyperparameters
            hparams = {
                'seed': seed,
                'model': 'MPO2',
                **config
            }
            tracker.log_hparams(hparams)
            
            # Log initial metrics (train and val)
            metrics = {
                'train_loss': scores_train['loss'],
                'train_quality': compute_quality(scores_train),
                'val_loss': scores_val['loss'],
                'val_quality': compute_quality(scores_val)
            }
            tracker.log_metrics(metrics, step=-1)  # step=-1 for init
    
    def callback_epoch(epoch, scores_train, scores_val, info):
        if tracker:
            # Log metrics for this epoch (train and val)
            metrics = {
                'train_loss': scores_train['loss'],
                'train_quality': compute_quality(scores_train),
                'val_loss': scores_val['loss'],
                'val_quality': compute_quality(scores_val),
                'reg_loss': info['reg_loss'],
                'jitter': info['jitter']
            }
            
            if info['weight_norm_sq'] is not None:
                metrics['weight_norm_sq'] = info['weight_norm_sq']
            
            tracker.log_metrics(metrics, step=epoch)
    
    # Train (optimizing on train, selecting best on val)
    try:
        scores_train, scores_val = ntn.fit(
            n_epochs=config['n_epochs'],
            regularize=True,
            jitter=jitter_schedule,
            eval_metrics=eval_metrics,
            val_data=loader_val,  # Pass validation Inputs object
            test_data=loader_test,  # Store test data in NTN
            verbose=config['verbose'],
            callback_init=callback_init,
            callback_epoch=callback_epoch
        )
        
        # Final evaluation on test set using the best model
        scores_test = ntn.evaluate(eval_metrics, data_stream=loader_test)
        
        # Extract metrics
        train_loss = scores_train['loss']
        train_quality = compute_quality(scores_train)
        val_loss = scores_val['loss']
        val_quality = compute_quality(scores_val)
        test_loss = scores_test['loss']
        test_quality = compute_quality(scores_test)
        
        # Check if training was successful
        success = test_quality is not None and test_quality > 0
        
        result = {
            'seed': seed,
            'train_loss': float(train_loss),
            'train_quality': float(train_quality),
            'val_loss': float(val_loss),
            'val_quality': float(val_quality),
            'test_loss': float(test_loss),
            'test_quality': float(test_quality),
            'success': success
        }
        
        # Log summary if tracker provided
        if tracker:
            tracker.log_summary(result)
        
        return result
        
    except Exception as e:
        result = {
            'seed': seed,
            'train_loss': None,
            'train_quality': None,
            'val_loss': None,
            'val_quality': None,
            'test_loss': None,
            'test_quality': None,
            'success': False,
            'error': str(e)
        }
        
        if tracker:
            tracker.log_summary(result)
        
        return result


def main():
    """Main training loop."""
    # Parse configuration
    config = parse_env()
    
    print("="*70)
    print("MPO2 TRAINING - MULTI-SEED ANALYSIS")
    print("="*70)
    print(f"\nConfiguration:")
    for key, value in config.items():
        if key != 'seeds':
            print(f"  {key}: {value}")
    print(f"  seeds: {config['seeds']} (n={len(config['seeds'])})")
    
    # Load dataset
    print(f"\nLoading dataset: {config['dataset']}...")
    data, dataset_info = load_dataset(
        config['dataset'],
        seed=config['seeds'][0]  # Use first seed for dataset sampling
    )
    
    input_dim = data['X_train'].shape[1] + 1  # +1 for bias (added by create_inputs)
    output_dim = data['y_train'].shape[1] if data['y_train'].ndim > 1 else 1
    
    print(f"  Dataset: {dataset_info['name']}")
    print(f"  Train: {dataset_info['n_train']} samples")
    print(f"  Val: {dataset_info['n_val']} samples")
    print(f"  Test: {dataset_info['n_test']} samples")
    print(f"  Features: {dataset_info['n_features']} (+1 bias = {input_dim})")
    print(f"  Task: {dataset_info['task']}")
    
    # Validate task matches dataset
    if config['task'] != dataset_info['task']:
        print(f"\nWARNING: Config task '{config['task']}' doesn't match dataset task '{dataset_info['task']}'")
        print(f"Using dataset task: {dataset_info['task']}")
        config['task'] = dataset_info['task']
    
    # Train with each seed
    results = []
    
    for seed_idx, seed in enumerate(config['seeds']):
        print(f"\nSeed {seed_idx + 1}/{len(config['seeds'])} (seed={seed})...")
        
        # Create tracker for this seed
        if config['tracker_backend'] != 'none':
            from experiments.trackers import create_tracker
            tracker = create_tracker(
                experiment_name=f"{config['experiment_name']}_seed{seed}",
                config=config,
                backend=config['tracker_backend'],
                output_dir=config['tracker_dir'],
                repo=config['aim_repo']
            )
        else:
            tracker = None
        
        # Train
        result = train_single_seed(config, seed, data, input_dim, output_dim, tracker)
        results.append(result)
        
        # Finalize tracker
        if tracker:
            tracker.finalize()
        
        # Print result (show test performance)
        if result['success']:
            quality_name = "R²" if config['task'] == 'regression' else "Acc"
            print(f"  ✓ Test: Loss={result['test_loss']:.6f}, {quality_name}={result['test_quality']:.6f} | Val: {quality_name}={result['val_quality']:.6f}")
        else:
            error_msg = result.get('error', 'Unknown error')
            print(f"  ✗ FAILED: {error_msg}")
    
    # Compute statistics (only successful runs, on TEST set)
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        test_losses = [r['test_loss'] for r in successful_results]
        test_qualities = [r['test_quality'] for r in successful_results]
        val_qualities = [r['val_quality'] for r in successful_results]
        
        stats = {
            'model': 'MPO2',
            'task': config['task'],
            'n_total': len(results),
            'n_success': len(successful_results),
            'test_loss_mean': float(np.mean(test_losses)),
            'test_loss_std': float(np.std(test_losses)),
            'test_quality_mean': float(np.mean(test_qualities)),
            'test_quality_std': float(np.std(test_qualities)),
            'val_quality_mean': float(np.mean(val_qualities)),
            'val_quality_std': float(np.std(val_qualities)),
            'results': results,
            'config': config,
            'dataset_info': dataset_info
        }
    else:
        stats = {
            'model': 'MPO2',
            'task': config['task'],
            'n_total': len(results),
            'n_success': 0,
            'test_loss_mean': None,
            'test_loss_std': None,
            'test_quality_mean': None,
            'test_quality_std': None,
            'val_quality_mean': None,
            'val_quality_std': None,
            'results': results,
            'config': config,
            'dataset_info': dataset_info
        }
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Model: MPO2")
    print(f"Task: {config['task']}")
    print(f"Successful runs: {stats['n_success']}/{stats['n_total']}")
    
    if stats['n_success'] > 0:
        quality_name = "R²" if config['task'] == 'regression' else "Accuracy"
        print(f"Test Loss: {stats['test_loss_mean']:.4f} ± {stats['test_loss_std']:.4f}")
        print(f"Test {quality_name}: {stats['test_quality_mean']:.4f} ± {stats['test_quality_std']:.4f}")
        print(f"Val {quality_name}: {stats['val_quality_mean']:.4f} ± {stats['val_quality_std']:.4f}")
    else:
        print("No successful runs!")
    
    # Save results
    with open(config['output_file'], 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nResults saved to: {config['output_file']}")
    print("="*70)


if __name__ == '__main__':
    main()
