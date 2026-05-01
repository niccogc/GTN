#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path

# Constants
NTN_MODELS = ["cpda", "cpda_typei", "lmpo2", "lmpo2_typei", "mpo2", "mpo2_typei", "mmpo2", "mmpo2_typei", "tnml_f", "tnml_p"]
DATASETS = ["abalone", "adult", "ai4i", "appliances", "bank", "bike", "breast", "car_evaluation", "concrete", "energy_efficiency", "hearth", "iris", "mushrooms", "obesity", "popularity", "realstate", "seoulBike", "student_dropout", "student_perf", "wine", "winequalityc"]
L_VALUES = [3, 4]
DEFAULT_BOND_DIMS = [4, 8, 12]
CPDA_BOND_DIMS = [8, 16, 32]
SEEDS = [42, 10090, 32874, 47311, 47303]

MODEL_NAME_MAP = {
    "cpda": "CPDA", "cpda_typei": "CPDATypeI", "lmpo2": "LMPO2", "lmpo2_typei": "LMPO2TypeI",
    "mpo2": "MPO2", "mpo2_typei": "MPO2TypeI", "mmpo2": "MMPO2", "mmpo2_typei": "MMPO2TypeI",
    "tnml_f": "TNML_F", "tnml_p": "TNML_P"
}

# Template for the Job Array script (submit_all.sh)
ARRAY_SCRIPT_TEMPLATE = '''#!/bin/bash
#BSUB -q hpc
#BSUB -J "ntn_{model}_{dataset}[1-{num_jobs}]%10"
#BSUB -W 1:00
#BSUB -n 6
#BSUB -R "rusage[mem=6GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/%J_%I.out
#BSUB -e logs/%J_%I.err

export HOME=/zhome/6b/e/212868
cd "$HOME/GTN"
source .venv/bin/activate
mkdir -p logs

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

# List of missing configs: "L bond_dim seed"
CONFIGS=(
{configs_list}
)

IDV=$((LSB_JOBINDEX - 1))
CONF=(${{CONFIGS[$IDV]}})

L=${{CONF[0]}}
BD=${{CONF[1]}}
SEED=${{CONF[2]}}

echo "Running: model={model_config}, dataset={dataset}, L=$L, bd=$BD, seed=$SEED"

python run.py \\
    model={model_config} \\
    dataset={dataset} \\
    model.L=$L \\
    model.bond_dim=$BD \\
    seed=$SEED
'''

def is_run_complete(model_config: str, dataset: str, L: int, bd: int, seed: int) -> bool:
    model_name = MODEL_NAME_MAP[model_config]
    path = Path("outputs") / "ntn" / dataset / f"{model_name}_rg5_init0.1" / f"L{L}_bd{bd}_seed{seed}" / "results.json"
    
    if not path.exists():
        return False
    try:
        with open(path) as f:
            data = json.load(f)
        return not data.get("oom_error", False)
    except:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("submit_individual_ntn"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total_jobs_generated = 0
    
    for model in NTN_MODELS:
        current_bond_dims = CPDA_BOND_DIMS if "cpda" in model else DEFAULT_BOND_DIMS
        
        for dataset in DATASETS:
            missing_configs = []
            
            # Identify all missing (L, bd, seed) for this model/dataset
            for L in L_VALUES:
                for bd in current_bond_dims:
                    for seed in SEEDS:
                        if not is_run_complete(model, dataset, L, bd, seed):
                            missing_configs.append(f'"{L} {bd} {seed}"')

            if not missing_configs:
                continue

            # Prep directory
            dataset_model_dir = args.output_dir / model / dataset
            if not args.dry_run:
                dataset_model_dir.mkdir(parents=True, exist_ok=True)

            # Generate the Job Array script
            num_jobs = len(missing_configs)
            configs_str = "\n".join([f"    {c}" for c in missing_configs])
            
            script_content = ARRAY_SCRIPT_TEMPLATE.format(
                model=model.replace("_", ""),
                model_config=model,
                dataset=dataset,
                num_jobs=num_jobs,
                configs_list=configs_str
            )

            if not args.dry_run:
                script_path = dataset_model_dir / "submit_all.sh"
                script_path.write_text(script_content)
                script_path.chmod(0o755)
                total_jobs_generated += num_jobs

    if args.dry_run:
        print("Dry run complete.")
    else:
        print(f"Generated Job Array scripts in {args.output_dir}")
        print(f"Total individual runs to be executed: {total_jobs_generated}")

if __name__ == "__main__":
    main()
