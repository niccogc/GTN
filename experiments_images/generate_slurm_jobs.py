#!/usr/bin/env python3
"""
Generate Slurm job scripts for DTU Titans cluster for image experiments (CMPO2/CMPO3).
"""

import os
import json
import argparse
from pathlib import Path

DATASET_SIZES = {
    "MNIST": "medium",
    "FASHION_MNIST": "medium",
    "CIFAR10": "large",
    "CIFAR100": "large",
}

# Titans cluster queue config
# Available GPU architectures: Pascal, Turing, Volta, Ampere, Ada
# Recommended: <=4 CPUs, <=32GB RAM per GPU for fair resource utilization
QUEUE_CONFIG = {
    "small": {
        "partition": "titans",
        "time": "4:00:00",
        "mem": "16gb",
        "gpu_arch": "Ampere",
        "cpus": 2,
    },
    "medium": {
        "partition": "titans",
        "time": "12:00:00",
        "mem": "32gb",
        "gpu_arch": "Ampere",
        "cpus": 4,
    },
    "large": {
        "partition": "titans",
        "time": "24:00:00",
        "mem": "32gb",
        "gpu_arch": "Ampere",
        "cpus": 4,
    },
}

JOB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%J.out
#SBATCH --error=logs/{job_name}_%J.err
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --gres=gpu:{gpu_arch}:{gpu}
#SBATCH --mail-user={email}
#SBATCH --mail-type=END,FAIL
#SBATCH --partition={partition}
#SBATCH --export=ALL

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\\n"

SCRATCH=/scratch/$USER
if [[ ! -d $SCRATCH ]]; then
  mkdir $SCRATCH
fi

export HOME=/home/nicci

cd $HOME/GTN
source .venv/bin/activate

set -a
source $HOME/aim
set +a

python experiments_images/run_grid_search_cmpo2.py --config experiments_images/configs/{config_file} --output-dir results/{experiment_name}

echo "Done: $(date +%F-%R:%S)"
"""


def get_dataset_size(dataset_name: str) -> str:
    return DATASET_SIZES.get(dataset_name, "medium")


def generate_job_script(
    config_file: str,
    output_dir: Path,
    partition_override: str | None = None,
    time_override: str | None = None,
    gpu_arch_override: str | None = None,
    email: str = "nicci@dtu.dk",
) -> Path:
    with open(config_file) as f:
        config = json.load(f)

    dataset = config["dataset"]
    experiment_name = config["experiment_name"]
    model_type = config.get("model_type", "cmpo2")

    size = get_dataset_size(dataset)

    # CMPO3 models need more resources
    if model_type == "cmpo3":
        if size == "medium":
            size = "large"
        mem_value = int(QUEUE_CONFIG[size]["mem"].replace("gb", ""))
        QUEUE_CONFIG[size]["mem"] = f"{int(mem_value * 1.5)}gb"

    queue_config = QUEUE_CONFIG[size].copy()

    if partition_override:
        queue_config["partition"] = partition_override
    if time_override:
        queue_config["time"] = time_override
    if gpu_arch_override:
        queue_config["gpu_arch"] = gpu_arch_override

    job_name = experiment_name.replace("_", "-")

    script_content = JOB_TEMPLATE.format(
        partition=queue_config["partition"],
        job_name=job_name,
        time=queue_config["time"],
        mem=queue_config["mem"],
        gpu_arch=queue_config["gpu_arch"],
        gpu=1,
        cpus=queue_config["cpus"],
        config_file=Path(config_file).name,
        experiment_name=experiment_name,
        email=email,
    )

    script_name = f"job_{experiment_name}.sh"
    script_path = output_dir / script_name

    with open(script_path, "w") as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)
    return script_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate Slurm job scripts for Titans cluster (image experiments)"
    )
    parser.add_argument(
        "--configs-dir",
        default="experiments_images/configs",
        help="Directory containing config files",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments_images/titans_jobs",
        help="Output directory for job scripts",
    )
    parser.add_argument("--partition", help="Override partition for all jobs (default: titans)")
    parser.add_argument("--time", help="Override time limit for all jobs (e.g., 24:00:00)")
    parser.add_argument(
        "--gpu-arch",
        choices=["Pascal", "Turing", "Volta", "Ampere", "Ada"],
        help="Override GPU architecture for all jobs",
    )
    parser.add_argument("--email", default="nicci@dtu.dk", help="Email for job notifications")
    parser.add_argument("--pattern", default="cmpo*.json", help="Config file pattern to match")
    args = parser.parse_args()

    configs_dir = Path(args.configs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "logs").mkdir(exist_ok=True)

    config_files = sorted(configs_dir.glob(args.pattern))

    config_files = [f for f in config_files if "test" not in f.name]

    print(f"Found {len(config_files)} config files")

    scripts = []
    for config_file in config_files:
        script_path = generate_job_script(
            str(config_file),
            output_dir,
            args.partition,
            args.time,
            args.gpu_arch,
            args.email,
        )
        scripts.append(script_path)
        print(f"  Created: {script_path.name}")

    submit_all = output_dir / "submit_all.sh"
    with open(submit_all, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all image experiment jobs to Slurm\n\n")
        f.write("mkdir -p logs\n\n")
        for script in scripts:
            f.write(f"sbatch {script.name}\n")
    os.chmod(submit_all, 0o755)

    print(f"\nGenerated {len(scripts)} job scripts in {output_dir}")
    print(f"Submit all with: cd {output_dir} && ./submit_all.sh")


if __name__ == "__main__":
    main()
