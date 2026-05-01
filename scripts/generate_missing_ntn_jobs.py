#!/usr/bin/env python3
"""Generate individual submit scripts for missing NTN model x dataset combinations.

Reads the tracking CSV to determine which combinations are complete,
and generates individual .sh files only for missing ones.

Usage:
    python scripts/generate_missing_ntn_jobs.py
    python scripts/generate_missing_ntn_jobs.py --dry-run
    python scripts/generate_missing_ntn_jobs.py --output-dir submit_cpu_ntn_missing
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

# Model configurations - maps config name to model class name in tracking
MODEL_CONFIG_TO_NAME = {
    "cpda": "CPDA",
    "cpda_typei": "CPDATypeI",
    "lmpo2": "LMPO2",
    "lmpo2_typei": "LMPO2TypeI",
    "mpo2": "MPO2",
    "mpo2_typei": "MPO2TypeI",
    "mmpo2": "MMPO2",
    "mmpo2_typei": "MMPO2TypeI",
    "tnml_f": "TNML_F",
    "tnml_p": "TNML_P",
}

# Reverse mapping
MODEL_NAME_TO_CONFIG = {v: k for k, v in MODEL_CONFIG_TO_NAME.items()}

# All NTN models (BosonMPS is GTN-only)
NTN_MODELS = [
    "cpda", "cpda_typei", "lmpo2", "lmpo2_typei",
    "mpo2", "mpo2_typei", "mmpo2", "mmpo2_typei",
    "tnml_f", "tnml_p"
]

# All datasets
DATASETS = [
    "abalone", "adult", "ai4i", "appliances", "bank", "bike", "breast",
    "car_evaluation", "concrete", "energy_efficiency", "hearth",
    "iris", "mushrooms", "obesity", "popularity", "realstate",
    "seoulBike", "student_dropout", "student_perf", "wine", "winequalityc"
]

# Expected runs per model
# Regular models: 2 L values * 3 bond_dims * 5 seeds = 30
# LMPO2 variants: 2 L * 3 bond_dims * 5 seeds * 3 reduction_factors = 90
# CPDA variants: 2 L * 4 bond_dims * 5 seeds = 40
EXPECTED_RUNS = {
    "CPDA": 40,
    "CPDATypeI": 40,
    "LMPO2": 90,
    "LMPO2TypeI": 90,
    "MPO2": 30,
    "MPO2TypeI": 30,
    "MMPO2": 30,
    "MMPO2TypeI": 30,
    "TNML_F": 30,
    "TNML_P": 30,
}

SCRIPT_TEMPLATE = '''#!/bin/bash
#BSUB -q hpc
#BSUB -J "ntn_{model_lower}_{dataset}"
#BSUB -W 8:00
#BSUB -n 6
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/%J_{model_lower}_{dataset}.out
#BSUB -e logs/%J_{model_lower}_{dataset}.err

export HOME=/zhome/6b/e/212868
cd "$HOME/GTN"
source .venv/bin/activate

mkdir -p logs

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

echo "Running: model={model_config}, dataset={dataset}, experiment={experiment}"

python run.py --multirun \\
    +experiment={experiment} \\
    model={model_config} \\
    dataset={dataset}
'''


def load_tracking_data(tracking_file: Path) -> dict[tuple[str, str], int]:
    """Load tracking CSV and count runs per (model, dataset) for NTN trainer.
    
    Returns:
        Dict mapping (model_name, dataset) -> count of completed runs
    """
    counts: dict[tuple[str, str], int] = defaultdict(int)
    
    with open(tracking_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["trainer_type"] != "ntn":
                continue
            model = row["model"]
            dataset = row["dataset"]
            counts[(model, dataset)] += 1
    
    return counts


def get_experiment_name(model_config: str) -> str:
    """Get the experiment name for a model config."""
    if model_config in ("cpda", "cpda_typei"):
        return "cpda_ntn_sweep"
    return "uci_ntn_sweep"


def generate_script(model_config: str, dataset: str) -> str:
    """Generate the shell script content for a model x dataset combination."""
    experiment = get_experiment_name(model_config)
    return SCRIPT_TEMPLATE.format(
        model_config=model_config,
        model_lower=model_config.replace("_", ""),
        dataset=dataset,
        experiment=experiment,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate submit scripts for missing NTN model x dataset combinations"
    )
    parser.add_argument(
        "--tracking-file",
        type=Path,
        default=Path("runs_tracking.csv"),
        help="Path to tracking CSV (default: runs_tracking.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("submit_cpu_ntn_missing"),
        help="Output directory for submit scripts (default: submit_cpu_ntn_missing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing files",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed progress info",
    )
    args = parser.parse_args()

    if not args.tracking_file.exists():
        print(f"Error: Tracking file not found: {args.tracking_file}")
        return 1

    # Load completed runs from tracking
    print(f"Loading tracking data from {args.tracking_file}...")
    completed = load_tracking_data(args.tracking_file)
    
    # Find missing combinations
    missing = []
    complete = []
    partial = []
    
    for model_config in NTN_MODELS:
        model_name = MODEL_CONFIG_TO_NAME[model_config]
        expected = EXPECTED_RUNS[model_name]
        
        for dataset in DATASETS:
            count = completed.get((model_name, dataset), 0)
            
            if count == 0:
                missing.append((model_config, dataset, 0, expected))
            elif count < expected:
                partial.append((model_config, dataset, count, expected))
            else:
                complete.append((model_config, dataset, count, expected))

    # Summary
    print(f"\nSummary:")
    print(f"  Complete:  {len(complete)} combinations")
    print(f"  Partial:   {len(partial)} combinations (have some runs)")
    print(f"  Missing:   {len(missing)} combinations (no runs at all)")
    
    # Combine partial and missing for generation
    to_generate = missing + partial
    print(f"\n  Total to generate: {len(to_generate)} scripts")
    
    if args.verbose:
        print("\nPartial combinations:")
        for model, dataset, count, expected in sorted(partial):
            print(f"  {model:15s} x {dataset:18s}: {count:3d}/{expected}")
        
        print("\nMissing combinations:")
        for model, dataset, count, expected in sorted(missing):
            print(f"  {model:15s} x {dataset:18s}: {count:3d}/{expected}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would generate the following scripts:")
        for model_config, dataset, count, expected in sorted(to_generate):
            script_path = args.output_dir / model_config / f"{dataset}.sh"
            print(f"  {script_path}  ({count}/{expected} complete)")
        return 0
    
    # Create output directory structure and generate scripts
    print(f"\nGenerating scripts in {args.output_dir}/...")
    
    generated_count = 0
    for model_config, dataset, count, expected in to_generate:
        model_dir = args.output_dir / model_config
        model_dir.mkdir(parents=True, exist_ok=True)
        
        script_path = model_dir / f"{dataset}.sh"
        script_content = generate_script(model_config, dataset)
        
        script_path.write_text(script_content)
        script_path.chmod(0o755)  # Make executable
        generated_count += 1
        
        if args.verbose:
            print(f"  Created: {script_path}")
    
    # Create a master submit script
    master_script = args.output_dir / "submit_all.sh"
    master_content = "#!/bin/bash\n# Submit all missing NTN jobs\n\n"
    master_content += f"cd \"$(dirname \"$0\")\"\n\n"
    
    for model_config, dataset, _, _ in sorted(to_generate):
        master_content += f"bsub < {model_config}/{dataset}.sh\n"
    
    master_script.write_text(master_content)
    master_script.chmod(0o755)
    
    print(f"\nGenerated {generated_count} individual scripts")
    print(f"Master script: {master_script}")
    print(f"\nTo submit all jobs: cd {args.output_dir} && bash submit_all.sh")
    
    return 0


if __name__ == "__main__":
    exit(main())
