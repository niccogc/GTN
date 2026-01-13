# type: ignore
"""
Grid search experiment runner.
Reads JSON configuration and runs all parameter combinations with tracking.
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from experiments.config_parser import (
    load_config, 
    create_experiment_plan, 
    print_experiment_summary
)
from experiments.dataset_loader import load_dataset
from experiments.trackers import create_tracker

from model.NTN import NTN
from model.losses import MSELoss, CrossEntropyLoss
from model.utils import REGRESSION_METRICS, CLASSIFICATION_METRICS, compute_quality, create_inputs
from model.MPO2_models import MPO2, LMPO2, MMPO2

torch.set_default_dtype(torch.float64)


def get_result_filepath(output_dir: str, run_id: str) -> str:
    """Get filepath for individual run result."""
    return os.path.join(output_dir, f"{run_id}.json")


def run_already_completed(output_dir: str, run_id: str) -> bool:
    """Check if a run has already been completed."""
    result_file = get_result_filepath(output_dir, run_id)
    
    if not os.path.exists(result_file):
        return False
    
    try:
        with open(result_file, 'r') as f:
            result = json.load(f)
        return result.get('success', False) is not None
    except:
        return False


def create_model(model_name: str, params: dict, input_dim: int, output_dim: int):
    """Create model instance based on model name and parameters."""
    
    if model_name == 'MPO2':
        return MPO2(
            L=params['L'],
            bond_dim=params['bond_dim'],
            phys_dim=input_dim,
            output_dim=output_dim,
            output_site=params.get('output_site', 1),
            init_strength=params.get('init_strength', 0.1)
        )
    
    elif model_name == 'LMPO2':
        return LMPO2(
            L=params['L'],
            bond_dim=params['bond_dim'],
            phys_dim=input_dim,
            output_dim=output_dim,
            rank=params.get('rank', 5),
            output_site=params.get('output_site', 1)
        )
    
    elif model_name == 'MMPO2':
        return MMPO2(
            L=params['L'],
            bond_dim=params['bond_dim'],
            phys_dim=input_dim,
            output_dim=output_dim,
            rank=params.get('rank', 5),
            output_site=params.get('output_site', 1)
        )
    
    elif model_name == 'MMPO2':
        return MMPO2(
            L=params['L'],
            bond_dim=params['bond_dim'],
            phys_dim=input_dim,
            output_dim=output_dim,
            rank=params['rank'],
            output_site=params.get('output_site', 1),
            init_strength=params.get('init_strength', 0.1)
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_single_experiment(experiment: dict, data: dict, input_dim: int, output_dim: int, 
                         verbose: bool = False, tracker=None):
    """Run a single experiment with given parameters."""
    
    params = experiment['params']
    seed = experiment['seed']
    task = experiment['task']
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if task == 'regression':
        loss_fn = MSELoss()
        eval_metrics = REGRESSION_METRICS
    else:
        loss_fn = CrossEntropyLoss()
        eval_metrics = CLASSIFICATION_METRICS
    
    n_epochs = params.get('n_epochs', 10)
    jitter_start = params.get('jitter_start', 0.001)
    jitter_decay = params.get('jitter_decay', 0.95)
    jitter_min = params.get('jitter_min', 0.001)
    
    jitter_schedule = [
        max(jitter_start * (jitter_decay ** epoch), jitter_min)
        for epoch in range(n_epochs)
    ]
    
    model_name = params['model']
    model = create_model(model_name, params, input_dim, output_dim)
    
    loader_train = create_inputs(
        X=data['X_train'], y=data['y_train'],
        input_labels=model.input_labels,
        output_labels=model.output_dims,
        batch_size=params.get('batch_size', 100),
        append_bias=False
    )
    
    loader_val = create_inputs(
        X=data['X_val'], y=data['y_val'],
        input_labels=model.input_labels,
        output_labels=model.output_dims,
        batch_size=params.get('batch_size', 100),
        append_bias=False
    )
    
    loader_test = create_inputs(
        X=data['X_test'], y=data['y_test'],
        input_labels=model.input_labels,
        output_labels=model.output_dims,
        batch_size=params.get('batch_size', 100),
        append_bias=False
    )
    
    ntn = NTN(
        tn=model.tn,
        output_dims=model.output_dims,
        input_dims=model.input_dims,
        loss=loss_fn,
        data_stream=loader_train
    )
    
    def callback_init(scores_train, scores_val, info):
        if tracker:
            hparams = {
                'seed': seed,
                'model': model_name,
                'dataset': experiment['dataset'],
                **params
            }
            tracker.log_hparams(hparams)
            
            metrics = {
                'train_loss': scores_train['loss'],
                'train_quality': compute_quality(scores_train),
                'val_loss': scores_val['loss'],
                'val_quality': compute_quality(scores_val)
            }
            tracker.log_metrics(metrics, step=-1)
    
    def callback_epoch(epoch, scores_train, scores_val, info):
        if tracker:
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
    
    try:
        scores_train, scores_val = ntn.fit(
            n_epochs=n_epochs,
            regularize=True,
            jitter=jitter_schedule,
            eval_metrics=eval_metrics,
            val_data=loader_val,
            test_data=loader_test,
            verbose=verbose,
            callback_init=callback_init,
            callback_epoch=callback_epoch,
            adaptive_jitter=params.get('adaptive_jitter', True),
            patience=params.get('patience', 5),
            min_delta=params.get('min_delta', 0.001),
            train_selection=params.get('train_selection', False)
        )
        
        scores_test = ntn.evaluate(eval_metrics, data_stream=loader_test)
        
        train_loss = scores_train['loss']
        train_quality = compute_quality(scores_train)
        val_loss = scores_val['loss']
        val_quality = compute_quality(scores_val)
        test_loss = scores_test['loss']
        test_quality = compute_quality(scores_test)
        
        success = test_quality is not None and test_quality > 0
        
        result = {
            'run_id': experiment['run_id'],
            'seed': seed,
            'model': model_name,
            'grid_params': experiment['grid_params'],
            'train_loss': float(train_loss),
            'train_quality': float(train_quality),
            'val_loss': float(val_loss),
            'val_quality': float(val_quality),
            'test_loss': float(test_loss),
            'test_quality': float(test_quality),
            'success': success
        }
        
        if tracker:
            tracker.log_summary(result)
        
        return result
        
    except Exception as e:
        result = {
            'run_id': experiment['run_id'],
            'seed': seed,
            'model': model_name,
            'grid_params': experiment['grid_params'],
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
    parser = argparse.ArgumentParser(description='Run grid search experiments')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to JSON configuration file')
    parser.add_argument('--resume', action='store_true',
                       help='Resume experiment (skip completed runs)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show experiment plan without running')
    parser.add_argument('--verbose', action='store_true',
                       help='Print training progress for each run')
    parser.add_argument('--aim-repo', type=str, default=None,
                       help='AIM repository URL (e.g., aim://192.168.5.5:5800 for VPN, aim://aimtracking.kosmon.org:443 for non-VPN)')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    experiments, metadata = create_experiment_plan(config)
    
    print_experiment_summary(experiments, metadata)
    
    if args.dry_run:
        print("Dry run complete. No experiments executed.")
        return
    
    output_dir = config['output']['results_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading dataset: {config['dataset']}...")
    data, dataset_info = load_dataset(config['dataset'])
    
    input_dim = data['X_train'].shape[1]
    output_dim = data['y_train'].shape[1] if data['y_train'].ndim > 1 else 1
    
    print(f"  Dataset: {dataset_info['name']}")
    print(f"  Train: {dataset_info['n_train']} samples")
    print(f"  Val: {dataset_info['n_val']} samples")
    print(f"  Test: {dataset_info['n_test']} samples")
    print(f"  Features: {dataset_info['n_features']} (+1 bias = {input_dim})")
    print(f"  Task: {dataset_info['task']}")
    print()
    
    if args.resume:
        completed = [exp for exp in experiments 
                    if run_already_completed(output_dir, exp['run_id'])]
        experiments = [exp for exp in experiments 
                      if not run_already_completed(output_dir, exp['run_id'])]
        print(f"Resume mode: Skipping {len(completed)} completed runs")
        print(f"Remaining: {len(experiments)} experiments to run")
        print()
    
    results = []
    start_time = time.time()
    
    aim_repo = args.aim_repo or os.getenv('AIM_REPO') or config['tracker'].get('aim_repo') or 'aim://aimtracking.kosmon.org:443'
    
    for i, experiment in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] Running: {experiment['run_id']}")
        
        if config['tracker']['backend'] != 'none':
            tracker = create_tracker(
                experiment_name=experiment['experiment_name'],
                config=experiment['params'],
                backend=config['tracker']['backend'],
                output_dir=config['tracker'].get('tracker_dir', 'experiment_logs'),
                repo=aim_repo,
                run_name=experiment['run_name']
            )
        else:
            tracker = None
        
        result = run_single_experiment(
            experiment, data, input_dim, output_dim,
            verbose=args.verbose,
            tracker=tracker
        )
        
        if tracker:
            tracker.finalize()
        
        results.append(result)
        
        if config['output']['save_individual_runs']:
            result_file = get_result_filepath(output_dir, experiment['run_id'])
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
        
        if result['success']:
            quality_name = "R²" if experiment['task'] == 'regression' else "Acc"
            print(f"  ✓ Test: {quality_name}={result['test_quality']:.4f} | Val: {quality_name}={result['val_quality']:.4f}")
        else:
            error_msg = result.get('error', 'Unknown error')
            print(f"  ✗ FAILED: {error_msg}")
    
    elapsed_time = time.time() - start_time
    
    successful_results = [r for r in results if r['success']]
    
    print("\n" + "="*70)
    print("GRID SEARCH COMPLETE")
    print("="*70)
    print(f"Total runs: {len(results)}")
    print(f"Successful runs: {len(successful_results)}")
    print(f"Failed runs: {len(results) - len(successful_results)}")
    if len(results) > 0:
        print(f"Time elapsed: {elapsed_time:.1f}s ({elapsed_time/len(results):.1f}s per run)")
    else:
        print(f"Time elapsed: {elapsed_time:.1f}s (all runs already completed)")
    print()
    
    if successful_results:
        quality_name = "R²" if config.get('task', 'regression') == 'regression' else "Accuracy"
        
        results_sorted = sorted(successful_results, 
                               key=lambda x: x['test_quality'], 
                               reverse=True)
        
        print(f"Top 5 Runs (by test {quality_name}):")
        print()
        for i, result in enumerate(results_sorted[:5]):
            print(f"{i+1}. {result['run_id']}")
            print(f"   Test {quality_name}: {result['test_quality']:.4f}")
            print(f"   Val {quality_name}: {result['val_quality']:.4f}")
            print(f"   Params: {result['grid_params']}")
            print()
    elif len(results) == 0 and args.resume:
        print("All experiments already completed. Check results directory for existing results.")
        print()
    
    summary = {
        'experiment_name': config['experiment_name'],
        'dataset': config['dataset'],
        'task': config.get('task', 'regression'),
        'total_experiments': len(results),
        'successful': len(successful_results),
        'failed': len(results) - len(successful_results),
        'elapsed_time': elapsed_time,
        'results': results,
        'metadata': metadata,
        'config': config
    }
    
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_file}")
    print("="*70)


if __name__ == '__main__':
    main()
