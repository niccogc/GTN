# type: ignore
"""
Flexible experiment tracking system supporting multiple backends.

Supported backends:
- 'file': Save metrics to JSON file (epoch-by-epoch)
- 'aim': Track with AIM experiment tracking
- 'none': No tracking (useful for quick tests)
"""
import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path


class BaseTracker:
    """Base class for all trackers."""
    
    def __init__(self, experiment_name: str, config: Dict[str, Any]):
        self.experiment_name = experiment_name
        self.config = config
    
    def log_hparams(self, hparams: Dict[str, Any]):
        """Log hyperparameters."""
        pass
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics for a specific step/epoch."""
        pass
    
    def log_summary(self, summary: Dict[str, Any]):
        """Log final summary statistics."""
        pass
    
    def finalize(self):
        """Cleanup and finalize tracking."""
        pass


class FileTracker(BaseTracker):
    """Track experiments to JSON files."""
    
    def __init__(self, experiment_name: str, config: Dict[str, Any], 
                 output_dir: str = "experiment_logs"):
        super().__init__(experiment_name, config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage
        self.hparams = {}
        self.metrics_log = []  # List of {step, metrics} dicts
        self.summary = {}
        
    def log_hparams(self, hparams: Dict[str, Any]):
        """Log hyperparameters."""
        self.hparams.update(hparams)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics for a specific step/epoch."""
        entry = {'step': step, **metrics}
        self.metrics_log.append(entry)
    
    def log_summary(self, summary: Dict[str, Any]):
        """Log final summary statistics."""
        self.summary.update(summary)
    
    def finalize(self):
        """Save all data to JSON file."""
        output_file = self.output_dir / f"{self.experiment_name}.json"
        
        data = {
            'experiment_name': self.experiment_name,
            'config': self.config,
            'hparams': self.hparams,
            'metrics_log': self.metrics_log,
            'summary': self.summary
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  Experiment log saved to: {output_file}")


class AIMTracker(BaseTracker):
    """Track experiments with AIM."""
    
    def __init__(self, experiment_name: str, config: Dict[str, Any],
                 repo: Optional[str] = None):
        super().__init__(experiment_name, config)
        
        # Import AIM
        try:
            # Try authenticated version first
            from aim_auth import Run
        except ImportError:
            # Fallback to regular AIM
            from aim import Run
        
        # Determine repo
        if repo is None:
            repo = os.getenv('AIM_REPO', '.aim')  # Default to local
        
        # Create run
        self.run = Run(
            repo=repo,
            experiment=experiment_name
        )
        
        # Log config
        self.run['config'] = config
    
    def log_hparams(self, hparams: Dict[str, Any]):
        """Log hyperparameters."""
        self.run['hparams'] = hparams
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics for a specific step/epoch."""
        for key, value in metrics.items():
            self.run.track(value, name=key, step=step)
    
    def log_summary(self, summary: Dict[str, Any]):
        """Log final summary statistics."""
        self.run['summary'] = summary
    
    def finalize(self):
        """Finalize AIM run."""
        self.run.close()
        print(f"  AIM run finalized: {self.run.hash}")


class MultiTracker(BaseTracker):
    """Track to multiple backends simultaneously."""
    
    def __init__(self, trackers: List[BaseTracker]):
        self.trackers = trackers
    
    def log_hparams(self, hparams: Dict[str, Any]):
        for tracker in self.trackers:
            tracker.log_hparams(hparams)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        for tracker in self.trackers:
            tracker.log_metrics(metrics, step)
    
    def log_summary(self, summary: Dict[str, Any]):
        for tracker in self.trackers:
            tracker.log_summary(summary)
    
    def finalize(self):
        for tracker in self.trackers:
            tracker.finalize()


class NoOpTracker(BaseTracker):
    """No-op tracker for when tracking is disabled."""
    pass


def create_tracker(experiment_name: str, config: Dict[str, Any],
                   backend: str = 'file', **kwargs) -> BaseTracker:
    """
    Factory function to create appropriate tracker.
    
    Args:
        experiment_name: Name of experiment
        config: Configuration dict
        backend: 'file', 'aim', 'both', or 'none'
        **kwargs: Additional arguments for specific trackers
            - output_dir: For file tracker
            - repo: For AIM tracker
    
    Returns:
        Tracker instance
    """
    if backend == 'file':
        return FileTracker(
            experiment_name, 
            config,
            output_dir=kwargs.get('output_dir', 'experiment_logs')
        )
    
    elif backend == 'aim':
        return AIMTracker(
            experiment_name,
            config,
            repo=kwargs.get('repo', None)
        )
    
    elif backend == 'both':
        file_tracker = FileTracker(
            experiment_name,
            config,
            output_dir=kwargs.get('output_dir', 'experiment_logs')
        )
        aim_tracker = AIMTracker(
            experiment_name,
            config,
            repo=kwargs.get('repo', None)
        )
        return MultiTracker([file_tracker, aim_tracker])
    
    elif backend == 'none':
        return NoOpTracker(experiment_name, config)
    
    else:
        raise ValueError(f"Unknown tracking backend: {backend}. "
                        f"Choose from: 'file', 'aim', 'both', 'none'")
