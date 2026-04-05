#!/usr/bin/env python3
"""
Generate cluster job scripts from Hydra configs.

Generates SLURM (Titans) and HPC (DTU LSF/bsub) job scripts organized by dataset size.

Output structure:
    submit/
    ├── hpc/
    │   ├── small/
    │   │   ├── job_ntn_mpo2_iris.sh
    │   │   └── submit.sh
    │   ├── medium/
    │   └── large/
    ├── slurm/
    │   ├── small/
    │   ├── medium/
    │   └── large/
    ├── submit_all.sh           # Submit everything
    ├── submit_all_hpc.sh       # All HPC jobs
    ├── submit_all_slurm.sh     # All SLURM jobs
    ├── submit_small_hpc.sh     # Small datasets on HPC
    ├── submit_medium_slurm.sh  # Medium datasets on SLURM
    └── ...

Usage:
    python scripts/generate_hydra_jobs.py                        # all datasets × all models
    python scripts/generate_hydra_jobs.py --cluster slurm        # SLURM only
    python scripts/generate_hydra_jobs.py --cluster hpc          # HPC only
    python scripts/generate_hydra_jobs.py --datasets iris,concrete
    python scripts/generate_hydra_jobs.py --models mpo2,lmpo2
    python scripts/generate_hydra_jobs.py --trainer gtn
    python scripts/generate_hydra_jobs.py --experiment uci_ntn_sweep
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf

# All available models
ALL_MODELS = ["mpo2", "lmpo2", "mmpo2", "mpo2_typei", "lmpo2_typei", "mmpo2_typei", "tnml_p", "tnml_f"]

# === Job Templates ===

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%J.out
#SBATCH --error=logs/{job_name}_%J.err
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --mem=2gb
#SBATCH --gres=gpu:{gpu_arch}:{gpu}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mail-user={email}
#SBATCH --mail-type=FAIL

echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"

export HOME=/home/nicci
cd $HOME/GTN
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gtn

set -a && source $HOME/aim && set +a

{command}

echo "Done: $(date +%F-%R:%S)"
"""

SLURM_ARRAY_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%A_%a.out
#SBATCH --error=logs/{job_name}_%A_%a.err
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --mem=2gb
#SBATCH --gres=gpu:{gpu_arch}:{gpu}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mail-user={email}
#SBATCH --mail-type=FAIL,ARRAY_TASKS
#SBATCH --array=1-{array_size}

echo "Node: $(hostname)"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Start: $(date +%F-%R:%S)"

export HOME=/home/nicci
cd $HOME/GTN
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gtn

set -a && source $HOME/aim && set +a

# Read parameters from params file (SLURM_SUBMIT_DIR is where sbatch was run from)
PARAMS_FILE="$SLURM_SUBMIT_DIR/{params_file}"
MODEL=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {{print $2}}' $PARAMS_FILE)
DATASET=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {{print $3}}' $PARAMS_FILE)

echo "Running: model=$MODEL dataset=$DATASET"

python run.py --multirun {experiment_arg} model=$MODEL dataset=$DATASET

echo "Done: $(date +%F-%R:%S)"
"""

HPC_TEMPLATE = """#!/bin/sh
#BSUB -q {queue}
#BSUB -J {job_name}
#BSUB -W {time}
#BSUB -n {cpus}
#BSUB -gpu "num={gpu}:mode=exclusive_process"
#BSUB -R "rusage[mem={mem}]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/{job_name}_%J.out
#BSUB -e logs/{job_name}_%J.err
#BSUB -u {email}

export HOME=/zhome/6b/e/212868
cd $HOME/GTN
source .venv/bin/activate

set -a && source $HOME/aim && set +a

{command}
"""


def get_dataset_size(dataset_name: str, conf_dir: Path) -> str:
    """Get size preset from dataset config."""
    dataset_file = conf_dir / "dataset" / f"{dataset_name}.yaml"
    if not dataset_file.exists():
        return "small"

    cfg = OmegaConf.load(dataset_file)
    if "defaults" in cfg:
        for default in cfg.defaults:
            if isinstance(default, str) and default.startswith("size/"):
                return default.split("/")[1]
    return "small"


def get_cluster_settings(size: str, cluster: str, conf_dir: Path) -> dict:
    """Load cluster settings from size preset."""
    size_file = conf_dir / "dataset" / "size" / f"{size}.yaml"
    if size_file.exists():
        cfg = OmegaConf.load(size_file)
        return OmegaConf.to_container(cfg).get(cluster, {})

    # Defaults
    if cluster == "slurm":
        return {
            "partition": "titans",
            "time": "24:00:00",
            "mem": "2gb",
            "gpu_arch": "Ampere",
            "gpu": 1,
            "cpus": 4,
        }
    else:
        return {"queue": "gpuv100", "time": "24:00", "mem": "500MB", "cpus": 4, "gpu": 1}


def get_all_datasets(conf_dir: Path) -> list[str]:
    """Get all dataset names from conf/dataset/*.yaml."""
    dataset_dir = conf_dir / "dataset"
    datasets = []
    for f in sorted(dataset_dir.glob("*.yaml")):
        if f.name.startswith("_"):
            continue
        datasets.append(f.stem)
    return datasets


def generate_array_job(
    jobs: list[tuple[str, str]],  # List of (model, dataset) tuples
    trainer: str,
    experiment: str | None,
    cluster: str,
    settings: dict,
    output_dir: Path,
    email: str,
    batch_name: str,
) -> Path:
    """Generate a SLURM array job script with parameter file."""
    if cluster != "slurm":
        raise ValueError("Array jobs are only supported for SLURM")

    # Create params file
    params_filename = f"params_{batch_name}.txt"
    params_path = output_dir / params_filename

    params_lines = ["TaskID\tModel\tDataset"]
    for idx, (model, dataset) in enumerate(jobs, start=1):
        params_lines.append(f"{idx}\t{model}\t{dataset}")
    params_path.write_text("\n".join(params_lines) + "\n")

    # Build experiment argument
    if experiment:
        experiment_arg = f"+experiment={experiment}"
    else:
        experiment_arg = f"trainer={trainer}"

    # Generate array script
    job_name = f"{trainer}-{batch_name}"
    script_content = SLURM_ARRAY_TEMPLATE.format(
        job_name=job_name,
        partition=settings.get("partition", "titans"),
        time=settings.get("time", "12:00:00"),
        gpu_arch=settings.get("gpu_arch", "Ampere"),
        gpu=settings.get("gpu", 1),
        cpus=settings.get("cpus", 4),
        email=email,
        array_size=len(jobs),
        params_file=params_filename,
        experiment_arg=experiment_arg,
    )

    script_path = output_dir / f"array_{batch_name}.sh"
    script_path.write_text(script_content)
    os.chmod(script_path, 0o755)

    return script_path


def generate_job(
    dataset: str,
    model: str,
    trainer: str,
    experiment: str | None,
    cluster: str,
    settings: dict,
    output_dir: Path,
    email: str,
) -> Path:
    """Generate a job script."""
    job_name = f"{trainer}_{model}_{dataset}"

    # Build hydra command
    if experiment:
        command = (
            f"python run.py --multirun +experiment={experiment} model={model} dataset={dataset}"
        )
    else:
        command = f"python run.py --multirun trainer={trainer} model={model} dataset={dataset}"

    if cluster == "slurm":
        script_content = SLURM_TEMPLATE.format(
            job_name=job_name.replace("_", "-"),
            partition=settings.get("partition", "titans"),
            time=settings.get("time", "12:00:00"),
            mem=settings.get("mem", "2gb"),
            gpu_arch=settings.get("gpu_arch", "Ampere"),
            gpu=settings.get("gpu", 1),
            cpus=settings.get("cpus", 4),
            email=email,
            command=command,
        )
    else:  # hpc
        script_content = HPC_TEMPLATE.format(
            job_name=job_name.replace("_", "-"),
            queue=settings.get("queue", "gpuv100"),
            time=settings.get("time", "12:00"),
            mem=settings.get("mem", "500MB"),
            gpu=settings.get("gpu", 1),
            cpus=settings.get("cpus", 4),
            email=email,
            command=command,
        )

    script_path = output_dir / f"job_{job_name}.sh"
    script_path.write_text(script_content)
    os.chmod(script_path, 0o755)
    return script_path


def create_submit_script(
    scripts: list[Path],
    output_path: Path,
    cluster: str,
    description: str,
) -> None:
    """Create a submit script that submits all given job scripts and records timestamp."""
    submit_cmd = "sbatch" if cluster == "slurm" else "bsub <"
    shebang = "#!/bin/bash" if cluster == "slurm" else "#!/bin/sh"

    # Calculate relative paths from submit script location to job scripts
    submit_dir = output_path.parent

    lines = [
        shebang,
        f"# {description}",
        f"# {len(scripts)} jobs",
        "",
        "# Record submission timestamp",
        "TIMESTAMP=$(date +%Y%m%d_%H%M%S)",
        "",
        "mkdir -p logs",
        "",
    ]

    for script in scripts:
        # Get relative path from submit script to job script
        try:
            rel_path = script.relative_to(submit_dir)
        except ValueError:
            # Scripts in different directory trees - use path from submit/
            rel_path = script
        lines.append(f"{submit_cmd} {rel_path}")

    # Get script name without extension for the marker file
    script_name = output_path.stem

    lines.extend(
        [
            "",
            "# Mark as submitted",
            f'echo "Submitted at $TIMESTAMP" > submitted_{script_name}_$TIMESTAMP',
            'echo "Submitted {len(scripts)} jobs at $(date)"'.replace(
                "{len(scripts)}", str(len(scripts))
            ),
        ]
    )

    output_path.write_text("\n".join(lines) + "\n")
    os.chmod(output_path, 0o755)


def main():
    parser = argparse.ArgumentParser(description="Generate cluster job scripts")
    parser.add_argument(
        "--conf-dir",
        type=Path,
        default=Path("conf"),
        help="Hydra config directory (default: conf)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for job scripts (default: submit_{trainer} or submit_{trainer}_slurm_array with --use-arrays)",
    )
    parser.add_argument(
        "--cluster",
        choices=["slurm", "hpc", "both"],
        default="both",
        help="Target cluster (default: both)",
    )
    parser.add_argument(
        "--trainer",
        choices=["ntn", "gtn"],
        default="ntn",
        help="Trainer type (default: ntn)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Use experiment config (e.g., uci_ntn_sweep)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of datasets (default: all)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=f"Comma-separated list of models (default: all = {','.join(ALL_MODELS)})",
    )
    parser.add_argument(
        "--email",
        default="nicci@dtu.dk",
        help="Email for job notifications",
    )
    parser.add_argument(
        "--use-arrays",
        action="store_true",
        help="Generate SLURM array jobs instead of individual jobs (SLURM only)",
    )
    parser.add_argument(
        "--array-size",
        type=int,
        default=6,
        help="Maximum number of jobs per array (default: 6)",
    )
    args = parser.parse_args()

    # Set default output directory based on mode
    if args.output_dir is None:
        if args.use_arrays:
            args.output_dir = Path(f"submit_{args.trainer}_slurm_array")
        else:
            args.output_dir = Path(f"submit_{args.trainer}")

    # Get datasets
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    else:
        datasets = get_all_datasets(args.conf_dir)

    # Get models
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = ALL_MODELS

    # Get clusters
    clusters = ["slurm", "hpc"] if args.cluster == "both" else [args.cluster]

    # Validate array job options
    if args.use_arrays:
        if "hpc" in clusters and "slurm" not in clusters:
            print("Error: Array jobs are only supported for SLURM, not HPC")
            return
        if args.cluster == "both":
            print("Warning: --use-arrays only generates SLURM jobs, skipping HPC")
            clusters = ["slurm"]

    total_jobs = len(datasets) * len(models) * len(clusters)
    print(
        f"Generating {len(datasets)} datasets × {len(models)} models × {len(clusters)} clusters = {total_jobs} jobs"
    )

    # Organize scripts by cluster and size
    # Structure: scripts_by_cluster_size[cluster][size] = [scripts]
    scripts_by_cluster_size: dict[str, dict[str, list[Path]]] = defaultdict(
        lambda: defaultdict(list)
    )
    all_scripts_by_cluster: dict[str, list[Path]] = defaultdict(list)

    if args.use_arrays:
        # For array mode, use flat structure: submit_{trainer}_slurm_array/{size}/
        for size in ["small", "medium", "large"]:
            size_dir = args.output_dir / size
            size_dir.mkdir(parents=True, exist_ok=True)
            (size_dir / "logs").mkdir(exist_ok=True)

            # Clean old scripts in this size directory
            for old_script in size_dir.glob("array_*.sh"):
                old_script.unlink()
            for old_params in size_dir.glob("params_*.txt"):
                old_params.unlink()
    else:
        # For individual job mode, use nested structure: submit_{trainer}/{cluster}/{size}/
        for cluster in clusters:
            for size in ["small", "medium", "large"]:
                size_dir = args.output_dir / cluster / size
                size_dir.mkdir(parents=True, exist_ok=True)
                (size_dir / "logs").mkdir(exist_ok=True)

                # Clean old scripts in this size directory
                for old_script in size_dir.glob("job_*.sh"):
                    old_script.unlink()

    if args.use_arrays:
        # === ARRAY JOB MODE ===
        # Group jobs by size first, then batch into arrays
        # Output structure: submit_{trainer}_slurm_array/{size}/
        jobs_by_size: dict[str, list[tuple[str, str]]] = defaultdict(list)

        for dataset in datasets:
            size = get_dataset_size(dataset, args.conf_dir)
            for model in models:
                jobs_by_size[size].append((model, dataset))

        cluster = "slurm"  # Arrays only support SLURM
        for size in ["small", "medium", "large"]:
            size_jobs = jobs_by_size[size]
            if not size_jobs:
                continue

            settings = get_cluster_settings(size, cluster, args.conf_dir)
            output_dir = args.output_dir / size  # Flat structure without cluster subdir

            # Split into batches of array_size
            batches = [
                size_jobs[i : i + args.array_size]
                for i in range(0, len(size_jobs), args.array_size)
            ]

            for batch_idx, batch in enumerate(batches):
                batch_name = f"{size}_{batch_idx + 1:02d}"
                script = generate_array_job(
                    jobs=batch,
                    trainer=args.trainer,
                    experiment=args.experiment,
                    cluster=cluster,
                    settings=settings,
                    output_dir=output_dir,
                    email=args.email,
                    batch_name=batch_name,
                )
                scripts_by_cluster_size[cluster][size].append(script)
                all_scripts_by_cluster[cluster].append(script)

            # Create per-size submit scripts in each size directory
            size_scripts = scripts_by_cluster_size[cluster][size]
            if size_scripts:
                size_dir = args.output_dir / size
                create_submit_script(
                    scripts=size_scripts,
                    output_path=size_dir / "submit.sh",
                    cluster=cluster,
                    description=f"Submit all {size} {args.trainer.upper()} array jobs to SLURM",
                )

        print(f"\nArray job mode: batching into groups of {args.array_size}")

    else:
        # === INDIVIDUAL JOB MODE (original behavior) ===
        for cluster in clusters:
            for dataset in datasets:
                size = get_dataset_size(dataset, args.conf_dir)
                settings = get_cluster_settings(size, cluster, args.conf_dir)
                output_dir = args.output_dir / cluster / size

                for model in models:
                    script = generate_job(
                        dataset=dataset,
                        model=model,
                        trainer=args.trainer,
                        experiment=args.experiment,
                        cluster=cluster,
                        settings=settings,
                        output_dir=output_dir,
                        email=args.email,
                    )
                    scripts_by_cluster_size[cluster][size].append(script)
                    all_scripts_by_cluster[cluster].append(script)

            # Create per-size submit scripts in each size directory
            for size in ["small", "medium", "large"]:
                size_scripts = scripts_by_cluster_size[cluster][size]
                if size_scripts:
                    size_dir = args.output_dir / cluster / size
                    create_submit_script(
                        scripts=size_scripts,
                        output_path=size_dir / "submit.sh",
                        cluster=cluster,
                        description=f"Submit all {size} {args.trainer.upper()} jobs to {cluster.upper()}",
                    )

    # Create top-level submit scripts
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_arrays:
        # === ARRAY MODE: Simplified structure ===
        cluster = "slurm"
        all_scripts = all_scripts_by_cluster[cluster]

        # Submit by size: submit_small.sh, submit_medium.sh, submit_large.sh
        for size in ["small", "medium", "large"]:
            size_scripts = scripts_by_cluster_size[cluster][size]
            if size_scripts:
                create_submit_script(
                    scripts=size_scripts,
                    output_path=args.output_dir / f"submit_{size}.sh",
                    cluster=cluster,
                    description=f"Submit {size} dataset {args.trainer.upper()} array jobs to SLURM",
                )

        # Submit all
        if all_scripts:
            submit_all_path = args.output_dir / "submit_all.sh"
            lines = [
                "#!/bin/bash",
                f"# Submit ALL {args.trainer.upper()} array jobs to SLURM",
                f"# {len(all_scripts)} array scripts",
                "",
                "# Record submission timestamp",
                "TIMESTAMP=$(date +%Y%m%d_%H%M%S)",
                "",
                "mkdir -p small/logs medium/logs large/logs",
                "",
            ]

            for script in all_scripts:
                rel_path = script.relative_to(args.output_dir)
                lines.append(f"sbatch {rel_path}")

            lines.extend(
                [
                    "",
                    "# Mark as submitted",
                    'echo "Submitted at $TIMESTAMP" > submitted_submit_all_$TIMESTAMP',
                    f'echo "Submitted {len(all_scripts)} array jobs at $(date)"',
                ]
            )

            submit_all_path.write_text("\n".join(lines) + "\n")
            os.chmod(submit_all_path, 0o755)

    else:
        # === INDIVIDUAL JOB MODE: Original structure ===
        # Submit all jobs for each cluster: submit_all_hpc.sh, submit_all_slurm.sh
        for cluster in clusters:
            if all_scripts_by_cluster[cluster]:
                create_submit_script(
                    scripts=all_scripts_by_cluster[cluster],
                    output_path=args.output_dir / f"submit_all_{cluster}.sh",
                    cluster=cluster,
                    description=f"Submit ALL {args.trainer.upper()} jobs to {cluster.upper()}",
                )

        # Submit by size across clusters: submit_small_hpc.sh, submit_medium_slurm.sh, etc.
        for cluster in clusters:
            for size in ["small", "medium", "large"]:
                size_scripts = scripts_by_cluster_size[cluster][size]
                if size_scripts:
                    create_submit_script(
                        scripts=size_scripts,
                        output_path=args.output_dir / f"submit_{size}_{cluster}.sh",
                        cluster=cluster,
                        description=f"Submit {size} dataset {args.trainer.upper()} jobs to {cluster.upper()}",
                    )

        # Submit all (everything)
        all_scripts = []
        for cluster in clusters:
            all_scripts.extend(all_scripts_by_cluster[cluster])

        if all_scripts:
            # Use first cluster's submit command style (or default to bash for mixed)
            primary_cluster = clusters[0] if len(clusters) == 1 else "slurm"

            # For "submit all" across both clusters, we need separate sections
            submit_all_path = args.output_dir / "submit_all.sh"
            lines = [
                "#!/bin/bash",
                f"# Submit ALL {args.trainer.upper()} jobs to all clusters",
                f"# {len(all_scripts)} total jobs",
                "",
                "# Record submission timestamp",
                "TIMESTAMP=$(date +%Y%m%d_%H%M%S)",
                "",
            ]

            for cluster in clusters:
                cluster_scripts = all_scripts_by_cluster[cluster]
                if cluster_scripts:
                    submit_cmd = "sbatch" if cluster == "slurm" else "bsub <"
                    lines.append(f"# === {cluster.upper()} jobs ===")
                    lines.append(
                        f"mkdir -p {cluster}/small/logs {cluster}/medium/logs {cluster}/large/logs"
                    )
                    lines.append("")
                    for script in cluster_scripts:
                        rel_path = script.relative_to(args.output_dir)
                        lines.append(f"{submit_cmd} {rel_path}")
                    lines.append("")

            lines.extend(
                [
                    "# Mark as submitted",
                    'echo "Submitted at $TIMESTAMP" > submitted_submit_all_$TIMESTAMP',
                    f'echo "Submitted {len(all_scripts)} jobs at $(date)"',
                ]
            )

            submit_all_path.write_text("\n".join(lines) + "\n")
            os.chmod(submit_all_path, 0o755)

    # Print summary
    print(f"\nGenerated job scripts in {args.output_dir}/")

    if args.use_arrays:
        print("\nArray job mode enabled:")
        print(f"  Max tasks per array: {args.array_size}")
        print("\nDirectory structure:")
        cluster = "slurm"
        for size in ["small", "medium", "large"]:
            n_scripts = len(scripts_by_cluster_size[cluster][size])
            if n_scripts > 0:
                # Count total tasks across arrays
                total_tasks = 0
                for script in scripts_by_cluster_size[cluster][size]:
                    params_file = script.parent / script.name.replace("array_", "params_").replace(
                        ".sh", ".txt"
                    )
                    if params_file.exists():
                        total_tasks += (
                            len(params_file.read_text().strip().split("\n")) - 1
                        )  # -1 for header
                print(f"  {size}/: {n_scripts} array scripts ({total_tasks} total tasks)")

        print("\nSubmit scripts:")
        all_scripts = all_scripts_by_cluster["slurm"]
        print(f"  submit_all.sh      - Submit all array jobs ({len(all_scripts)} arrays)")
        for size in ["small", "medium", "large"]:
            n = len(scripts_by_cluster_size[cluster][size])
            if n > 0:
                print(f"  submit_{size}.sh   - {size.capitalize()} arrays ({n})")
                print(f"  {size}/submit.sh   - (same, inside folder)")
    else:
        print("\nDirectory structure:")
        for cluster in clusters:
            for size in ["small", "medium", "large"]:
                n_scripts = len(scripts_by_cluster_size[cluster][size])
                if n_scripts > 0:
                    print(f"  {cluster}/{size}/: {n_scripts} jobs")

        all_scripts = []
        for cluster in clusters:
            all_scripts.extend(all_scripts_by_cluster[cluster])

        print("\nSubmit scripts:")
        print(f"  submit_all.sh                - Submit everything ({len(all_scripts)} jobs)")
        for cluster in clusters:
            n = len(all_scripts_by_cluster[cluster])
            print(f"  submit_all_{cluster}.sh         - All {cluster.upper()} jobs ({n})")
        for cluster in clusters:
            for size in ["small", "medium", "large"]:
                n = len(scripts_by_cluster_size[cluster][size])
                if n > 0:
                    print(
                        f"  submit_{size}_{cluster}.sh      - {size.capitalize()} {cluster.upper()} ({n})"
                    )


if __name__ == "__main__":
    main()
