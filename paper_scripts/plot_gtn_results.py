"""
Plot GTN experiment results: test_accuracy vs n_parameters, colored by dataset.
Data from experiments_imgs/outputs/gtn/{CIFAR10,CIFAR100,FASHION_MNIST,MNIST}
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

GTN_DIR = Path(__file__).parent.parent / "outputs" / "gtn"
IMAGES_DIR = Path(__file__).parent / "images"


def parse_run_dir_name(name: str) -> dict:
    """Parse run directory name like 'L3_bd4_seed42' or 'L4_bd12_seed10090_rf0.3_bondmpo1'."""
    parts = name.split("_")
    result = {}
    for part in parts:
        if part.startswith("L") and part[1:].isdigit():
            result["L"] = int(part[1:])
        elif part.startswith("bd") and part[2:].isdigit():
            result["bd"] = int(part[2:])
        elif part.startswith("seed"):
            try:
                result["seed"] = int(part[4:])
            except ValueError:
                pass
        elif part.startswith("rf"):
            try:
                result["rf"] = float(part[2:])
            except ValueError:
                pass
        elif part.startswith("rp") and part[2:].isdigit():
            result["rp"] = int(part[2:])
        elif part.startswith("rpa") and part[3:].isdigit():
            result["rpa"] = int(part[3:])
    return result


def make_plots():
    grouped = defaultdict(list)
    param_map = {}
    
    for dataset_dir in sorted(GTN_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for run_dir in sorted(model_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                results_file = run_dir / "results.json"
                if not results_file.exists():
                    continue
                with open(results_file) as f:
                    data = json.load(f)
                
                test_acc = data.get("test_accuracy")
                n_params = data.get("n_parameters")
                if test_acc is None or n_params is None:
                    continue
                
                run_info = parse_run_dir_name(run_dir.name)
                L = run_info.get("L", 0)
                bd = run_info.get("bd", 0)
                rp = run_info.get("rp", 0)
                rpa = run_info.get("rpa", 0)
                
                key = (dataset_name, model_dir.name, bd, L, rp, rpa)
                grouped[key].append(test_acc)
                param_map[key] = n_params

    datasets = defaultdict(list)
    for key, accs in grouped.items():
        dataset_name = key[0]
        n_params = param_map[key]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs) if len(accs) > 1 else 0.0
        datasets[dataset_name].append((n_params, mean_acc, std_acc))

    for ds in datasets:
        datasets[ds].sort(key=lambda x: x[0])

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']
    dataset_names = sorted(datasets.keys())

    for i, ds in enumerate(dataset_names):
        points = datasets[ds]
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        yerrs = np.array([p[2] for p in points])
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # Sort by x for proper fill_between
        sort_idx = np.argsort(xs)
        xs_sorted = xs[sort_idx]
        ys_sorted = ys[sort_idx]
        yerrs_sorted = yerrs[sort_idx]
        
        # Fill between for error surface
        ax.fill_between(xs_sorted, ys_sorted - yerrs_sorted, ys_sorted + yerrs_sorted,
                        color=color, alpha=0.2)
        
        # Scatter plot with error bars (no connecting line)
        ax.errorbar(xs, ys, yerr=yerrs, fmt=marker, capsize=2, capthick=0.5,
                    markersize=4, color=color, label=ds,
                    alpha=0.9, markeredgewidth=0.3, markeredgecolor="black",
                    linestyle='none')

    ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
    ax.set_xlabel("Number of Parameters", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("GTN Test Accuracy vs Number of Parameters", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(IMAGES_DIR / "gtn_test_accuracy_vs_nparams.pdf"), dpi=150)
    plt.close(fig)
    print(f"Saved: {IMAGES_DIR / 'gtn_test_accuracy_vs_nparams.pdf'}")

    if len(dataset_names) >= 4:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, max(1, len(dataset_names)), figsize=(6 * len(dataset_names), 6))
        if len(dataset_names) == 1:
            axes = [axes]

    for idx, ds in enumerate(dataset_names[:len(axes)]):
        ax = axes[idx]
        points = datasets[ds]
        l_groups = defaultdict(list)
        for n_params, mean_acc, std_acc in points:
            for key, accs in grouped.items():
                if key[0] == ds and param_map[key] == n_params:
                    l_val = key[3]
                    l_groups[l_val].append((n_params, mean_acc, std_acc))
                    break

        for l_idx, (l_val, pts) in enumerate(sorted(l_groups.items())):
            pts.sort(key=lambda x: x[0])
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            yerrs = np.array([p[2] for p in pts])
            color = colors[l_idx % len(colors)]
            marker = markers[l_idx % len(markers)]
            
            # Fill between for error surface
            ax.fill_between(xs, ys - yerrs, ys + yerrs, color=color, alpha=0.2)
            
            # Scatter plot with error bars (no connecting line)
            ax.errorbar(xs, ys, yerr=yerrs, fmt=marker, capsize=2, capthick=0.5,
                        markersize=4, color=color, label=f"L={l_val}",
                        alpha=0.9, markeredgewidth=0.3, markeredgecolor="black",
                        linestyle='none')

        ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
        ax.set_xlabel("Number of Parameters")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(ds)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("GTN Test Accuracy vs N Params by Dataset and L", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(IMAGES_DIR / "gtn_test_accuracy_by_L.pdf"), dpi=150)
    plt.close(fig)
    print(f"Saved: {IMAGES_DIR / 'gtn_test_accuracy_by_L.pdf'}")

    print("\n--- Summary ---")
    for ds in dataset_names:
        print(f"\n{ds}:")
        for key, accs in grouped.items():
            if key[0] == ds:
                n = param_map[key]
                mean = np.mean(accs)
                std = np.std(accs) if len(accs) > 1 else 0
                n_runs = len(accs)
                print(f"  L={key[3]:2d}  rp={key[4]:2d}  rpa={key[5]:2d}  n_params={n:6d}  acc={mean:.4f} ± {std:.4f}  (n={n_runs})")


if __name__ == "__main__":
    make_plots()
