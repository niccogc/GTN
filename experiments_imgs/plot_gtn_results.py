"""
Plot GTN experiment results: test_accuracy vs n_parameters, colored by dataset.
Data from experiments_imgs/outputs/gtn/{CIFAR10,CIFAR100,FASHION_MNIST,MNIST}
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

GTN_DIR = Path("/home/nicci/Desktop/remote/GTN/experiments_imgs/outputs/gtn")

def make_plots():
    datasets = defaultdict(list)
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
                if not data.get("success", False):
                    continue
                datasets[dataset_name].append((data["n_parameters"], data["test_accuracy"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    for i, ds in enumerate(sorted(datasets.keys())):
        points = sorted(datasets[ds], key=lambda x: x[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, "o-", markersize=4, linewidth=1.2, color=colors[i], label=ds, alpha=0.8)

    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("GTN Test Accuracy vs Number of Parameters")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = GTN_DIR.parent / "gtn_test_accuracy_vs_nparams.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

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
    dataset_names = sorted(datasets.keys())

    for i, ds in enumerate(dataset_names):
        points = datasets[ds]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        yerrs = [p[2] for p in points]
        color = colors[i % len(colors)]
        ax.errorbar(xs, ys, yerr=yerrs, fmt="o-", capsize=3, capthick=1,
                     markersize=5, linewidth=1.5, color=color, label=ds,
                     alpha=0.8, markeredgewidth=0.5, markeredgecolor="black")

    ax.set_xlabel("Number of Parameters", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("GTN Test Accuracy vs Number of Parameters", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(GTN_DIR.parent / "gtn_test_accuracy_vs_nparams.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {GTN_DIR.parent / 'gtn_test_accuracy_vs_nparams.png'}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, ds in enumerate(dataset_names):
        ax = axes[idx]
        points = datasets[ds]
        l_groups = defaultdict(list)
        for n_params, mean_acc, std_acc in points:
            for key, accs in grouped.items():
                if key[0] == ds and param_map[key] == n_params:
                    l_val = key[3]
                    l_groups[l_val].append((n_params, mean_acc, std_acc))
                    break

        for l_val, pts in sorted(l_groups.items()):
            pts.sort(key=lambda x: x[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, "o-", markersize=5, linewidth=1.5, label=f"L={l_val}")

        ax.set_xlabel("Number of Parameters")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(ds)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("GTN Test Accuracy vs N Params by Dataset and L", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(GTN_DIR.parent / "gtn_test_accuracy_by_L.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {GTN_DIR.parent / 'gtn_test_accuracy_by_L.png'}")

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
