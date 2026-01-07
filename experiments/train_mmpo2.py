# type: ignore
"""
Production training script for MMPO2 model with multi-seed analysis.
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
from model.losses import MSELoss
from model.utils import REGRESSION_METRICS, compute_final_r2, create_inputs
from model.MPO2_models import MMPO2
from experiments.dataset_loader import load_dataset

torch.set_default_dtype(torch.float32)


def parse_env():
    """Parse configuration from environment variables."""
    config = {
        # Task type
        'task': os.getenv('TASK', 'regression').lower(),  # 'regression' or 'classification'
        
        # Dataset
        'dataset': os.getenv('DATASET', 'california_housing'),
        'n_samples': int(os.getenv('N_SAMPLES', '500')),
        
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
        'output_file': os.getenv('OUTPUT_FILE', 'results_mmpo2.json'),
        'verbose': os.getenv('VERBOSE', 'false').lower() == 'true',
        
        # Tracking
        'tracker_backend': os.getenv('TRACKER_BACKEND', 'file'),
        'tracker_dir': os.getenv('TRACKER_DIR', 'experiment_logs'),
        'aim_repo': os.getenv('AIM_REPO', None),
        'experiment_name': os.getenv('EXPERIMENT_NAME', 'mmpo2_experiment')
    }
    
    # Validate task type
    if config['task'] not in ['regression', 'classification']:
        raise ValueError(f"TASK must be 'regression' or 'classification', got: {config['task']}")
    
    return config


def train_single_seed(config, seed, X, y, input_dim, output_dim, tracker=None):
    """Train model with a single seed."""
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Select loss function and metrics based on task
    if config['task'] == 'regression':
        from model.losses import MSELoss
        from model.utils import REGRESSION_METRICS
        loss_fn = MSELoss()
        eval_metrics = REGRESSION_METRICS
    else:  # classification
        from model.losses import CrossEntropyLoss
        from model.utils import CLASSIFICATION_METRICS
        loss_fn = CrossEntropyLoss()
        eval_metrics = CLASSIFICATION_METRICS
    
    # Create jitter schedule
    jitter_schedule = [
        max(config['jitter_start'] * (config['jitter_decay'] ** epoch), config['jitter_min'])
        for epoch in range(config['n_epochs'])
    ]
    
    # Create model
    model = MMPO2(
        L=config['L'],
        bond_dim=config['bond_dim'],
        input_dim=input_dim,
        output_dim=output_dim,
        output_site=config['output_site'],
        init_strength=config['init_strength']
    )
    
    # Create data loader
    loader = create_inputs(
        X=X, y=y,
        input_labels=model.input_labels,
        output_labels=model.output_dims,
        batch_size=config['batch_size']
    )
    
    # Create NTN wrapper
    ntn = NTN(
        tn=model.tn,
        output_dims=model.output_dims,
        input_dims=model.input_dims,
        loss=loss_fn,
        data_stream=loader
    )
    
    # Setup callbacks for tracking
    def callback_init(scores, info):
        if tracker:
            hparams = {
                'seed': seed,
                'model': 'MMPO2',
                **config
            }
            tracker.log_hparams(hparams)
            
            if config['task'] == 'regression':
                metrics = {
                    'loss': scores['loss'],
                    'r2': compute_final_r2(scores)
                }
            else:  # classification
                metrics = {
                    'loss': scores['loss'],
                    'accuracy': scores['accuracy']
                }
            tracker.log_metrics(metrics, step=-1)
    
    def callback_epoch(epoch, scores, info):
        if tracker:
            if config['task'] == 'regression':
                metrics = {
                    'loss': scores['loss'],
                    'r2': compute_final_r2(scores),
                    'reg_loss': info['reg_loss'],
                    'jitter': info['jitter']
                }
            else:  # classification
                metrics = {
                    'loss': scores['loss'],
                    'accuracy': scores['accuracy'],
                    'reg_loss': info['reg_loss'],
                    'jitter': info['jitter']
                }
            
            if info['weight_norm_sq'] is not None:
                metrics['weight_norm_sq'] = info['weight_norm_sq']
            
            tracker.log_metrics(metrics, step=epoch)
    
    # Train
    try:
        scores = ntn.fit(
            n_epochs=config['n_epochs'],
            regularize=True,
            jitter=jitter_schedule,
            eval_metrics=eval_metrics,
            verbose=config['verbose'],
            callback_init=callback_init,
            callback_epoch=callback_epoch
        )
        
        # Extract metrics (task-dependent)
        loss = scores['loss']
        
        if config['task'] == 'regression':
            r2 = compute_final_r2(scores)
            success = r2 > 0
            
            result = {
                'seed': seed,
                'loss': float(loss),
                'r2': float(r2),
                'success': success
            }
        else:  # classification
            accuracy = scores['accuracy']
            success = accuracy > 0
            
            result = {
                'seed': seed,
                'loss': float(loss),
                'accuracy': float(accuracy),
                'success': success
            }
        
        if tracker:
            tracker.log_summary(result)
        
        return result
        
    except Exception as e:
        if config['task'] == 'regression':
            result = {
                'seed': seed,
                'loss': None,
                'r2': None,
                'success': False,
                'error': str(e)
            }
        else:  # classification
            result = {
                'seed': seed,
                'loss': None,
                'accuracy': None,
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
    print("MMPO2 TRAINING - MULTI-SEED ANALYSIS")
    print("="*70)
    print(f"\nConfiguration:")
    for key, value in config.items():
        if key != 'seeds':
            print(f"  {key}: {value}")
    print(f"  seeds: {config['seeds']} (n={len(config['seeds'])})")
    
    # Load dataset
    print(f"\nLoading dataset: {config['dataset']}...")
    X, y, dataset_info = load_dataset(
        config['dataset'],
        n_samples=config['n_samples'],
        seed=config['seeds'][0]  # Use first seed for dataset sampling
    )
    
    input_dim = X.shape[1] + 1  # +1 for bias (added by create_inputs)
    output_dim = y.shape[1] if y.ndim > 1 else 1
    
    print(f"  Dataset: {dataset_info['name']}")
    print(f"  Samples: {dataset_info['n_samples']}")
    print(f"  Features: {dataset_info['n_features']} (+1 bias = {input_dim})")
    print(f"  Mask bond dim: {input_dim} (= input_dim)")
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
        
        result = train_single_seed(config, seed, X, y, input_dim, output_dim, tracker)
        results.append(result)
        
        # Finalize tracker
        if tracker:
            tracker.finalize()
        
        if result['success']:
            if config['task'] == 'regression':
                print(f"  ✓ Loss={result['loss']:.6f}, R²={result['r2']:.6f}")
            else:  # classification
                print(f"  ✓ Loss={result['loss']:.6f}, Accuracy={result['accuracy']:.6f}")
        else:
            error_msg = result.get('error', 'Unknown error')
            print(f"  ✗ FAILED: {error_msg}")
    
    # Compute statistics (only successful runs)
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        losses = [r['loss'] for r in successful_results]
        
        if config['task'] == 'regression':
            r2s = [r['r2'] for r in successful_results]
            
            stats = {
                'model': 'MMPO2',
                'task': config['task'],
                'n_total': len(results),
                'n_success': len(successful_results),
                'loss_mean': float(np.mean(losses)),
                'loss_std': float(np.std(losses)),
                'r2_mean': float(np.mean(r2s)),
                'r2_std': float(np.std(r2s)),
                'results': results,
                'config': config,
                'dataset_info': dataset_info
            }
        else:  # classification
            accuracies = [r['accuracy'] for r in successful_results]
            
            stats = {
                'model': 'MMPO2',
                'task': config['task'],
                'n_total': len(results),
                'n_success': len(successful_results),
                'loss_mean': float(np.mean(losses)),
                'loss_std': float(np.std(losses)),
                'accuracy_mean': float(np.mean(accuracies)),
                'accuracy_std': float(np.std(accuracies)),
                'results': results,
                'config': config,
                'dataset_info': dataset_info
            }
    else:
        stats = {
            'model': 'MMPO2',
            'task': config['task'],
            'n_total': len(results),
            'n_success': 0,
            'loss_mean': None,
            'loss_std': None,
            'results': results,
            'config': config,
            'dataset_info': dataset_info
        }
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Model: MMPO2")
    print(f"Task: {config['task']}")
    print(f"Successful runs: {stats['n_success']}/{stats['n_total']}")
    
    if stats['n_success'] > 0:
        print(f"Loss: {stats['loss_mean']:.4f} ± {stats['loss_std']:.4f}")
        if config['task'] == 'regression':
            print(f"R²:   {stats['r2_mean']:.4f} ± {stats['r2_std']:.4f}")
        else:  # classification
            print(f"Accuracy: {stats['accuracy_mean']:.4f} ± {stats['accuracy_std']:.4f}")
    else:
        print("No successful runs!")
    
    # Save results
    with open(config['output_file'], 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nResults saved to: {config['output_file']}")
    print("="*70)


if __name__ == '__main__':
    main()
