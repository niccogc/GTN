#!/usr/bin/env python3
"""
Generate LSF job scripts for DTU HPC cluster.

Creates .sh scripts for each UCI dataset experiment config.
"""

import os
import json
import argparse
from pathlib import Path

DATASET_SIZES = {
    "iris": "small",
    "wine": "small",
    "breast": "small",
    "hearth": "small",
    "car_evaluation": "small",
    "student_perf": "small",
    "realstate": "small",
    "concrete": "small",
    "energy_efficiency": "small",
    "abalone": "medium",
    "winequalityc": "medium",
    "obesity": "medium",
    "bike": "medium",
    "ai4i": "medium",
    "seoulBike": "medium",
    "appliances": "large",
    "popularity": "large",
    "adult": "large",
    "bank": "large",
    "student_dropout": "large",
    "mushrooms": "large",
}

QUEUE_CONFIG = {
    "small": {"queue": "gpuv100", "time": "3:00", "mem": "8GB", "gpu": 1},
    "medium": {"queue": "gpuv100", "time": "12:00", "mem": "16GB", "gpu": 1},
    "large": {"queue": "gpua100", "time": "24:00", "mem": "32GB", "gpu": 1},
}

JOB_TEMPLATE = """#!/bin/sh
#BSUB -q {queue}
#BSUB -J {job_name}
#BSUB -W {time}
#BSUB -n 8
#BSUB -gpu "num={gpu}:mode=exclusive_process"
#BSUB -R "rusage[mem={mem}]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/{job_name}_%J.out
#BSUB -e logs/{job_name}_%J.err
#BSUB -u nicci@dtu.dk

export HOME=/zhome/6b/e/212868
export AIM_REPO=$HOME/aim

cd $HOME/GTN
source .venv/bin/activate

set -a
source $HOME/.ssh/aim
set +a

python experiments/{runner} --config experiments/configs/{config_file} {extra_args}
"""


def get_dataset_size(dataset_name: str) -> str:
    return DATASET_SIZES.get(dataset_name, "medium")


def generate_job_script(
    config_file: str,
    runner: str,
    output_dir: Path,
    queue_override: str | None = None,
    time_override: str | None = None,
) -> Path:
    with open(config_file) as f:
        config = json.load(f)

    dataset = config["dataset"]
    experiment_name = config["experiment_name"]
    size = get_dataset_size(dataset)
    queue_config = QUEUE_CONFIG[size].copy()

    if queue_override:
        queue_config["queue"] = queue_override
    if time_override:
        queue_config["time"] = time_override

    is_gtn = "gtn" in config_file.lower()
    runner_script = "run_grid_search_gtn.py" if is_gtn else "run_grid_search.py"

    extra_args = ""
    if is_gtn:
        extra_args = f"--output-dir results/{experiment_name}"

    job_name = experiment_name.replace("_", "-")

    script_content = JOB_TEMPLATE.format(
        queue=queue_config["queue"],
        job_name=job_name,
        time=queue_config["time"],
        mem=queue_config["mem"],
        gpu=queue_config["gpu"],
        runner=runner_script,
        config_file=Path(config_file).name,
        extra_args=extra_args,
    )

    script_name = f"job_{experiment_name}.sh"
    script_path = output_dir / script_name

    with open(script_path, "w") as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)
    return script_path


def main():
    parser = argparse.ArgumentParser(description="Generate HPC job scripts")
    parser.add_argument(
        "--configs-dir",
        default="experiments/configs",
        help="Directory containing config files",
    )
    parser.add_argument(
        "--output-dir", default="experiments/hpc_jobs", help="Output directory for job scripts"
    )
    parser.add_argument("--queue", help="Override queue for all jobs")
    parser.add_argument("--time", help="Override time limit for all jobs (e.g., 2:00)")
    parser.add_argument("--pattern", default="uci_*.json", help="Config file pattern to match")
    args = parser.parse_args()

    configs_dir = Path(args.configs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "logs").mkdir(exist_ok=True)

    config_files = sorted(configs_dir.glob(args.pattern))
    print(f"Found {len(config_files)} config files")

    scripts = []
    for config_file in config_files:
        script_path = generate_job_script(
            str(config_file),
            "run_grid_search.py",
            output_dir,
            args.queue,
            args.time,
        )
        scripts.append(script_path)
        print(f"  Created: {script_path.name}")

    submit_all = output_dir / "submit_all.sh"
    with open(submit_all, "w") as f:
        f.write("#!/bin/sh\n")
        f.write("# Submit all UCI experiment jobs\n\n")
        f.write("mkdir -p logs\n\n")
        for script in scripts:
            f.write(f"bsub < {script}\n")
    os.chmod(submit_all, 0o755)

    print(f"\nGenerated {len(scripts)} job scripts in {output_dir}")
    print(f"Submit all with: cd {output_dir} && ./submit_all.sh")


if __name__ == "__main__":
    main()
