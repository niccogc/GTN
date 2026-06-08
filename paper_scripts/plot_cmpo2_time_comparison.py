#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent.parent / "outputs" / "time_comparison"
NTN_FILE = BASE_DIR / "ntn/MNIST/CMPO2_np4_rp10_seed42/results.json"
GTN_FILE = BASE_DIR / "gtn/MNIST/CMPO2_np4_rp10_seed42/results.json"
OUTPUT_PATH = Path(__file__).parent / "images" / "cmpo2_val_quality_vs_time.pdf"

with open(NTN_FILE) as f:
    ntn_data = json.load(f)

with open(GTN_FILE) as f:
    gtn_data = json.load(f)

gtn_metrics = gtn_data["metrics_log"]
gtn_times = [m["wall_time"] for m in gtn_metrics]
gtn_vals = [m["val_quality"]*100 for m in gtn_metrics]

gtn_init_val = gtn_vals[0]

ntn_metrics = [m for m in ntn_data["metrics_log"] if m["epoch"] >= 0]
ntn_metrics = ntn_metrics[:-1]
ntn_times = [0.0] + [m["wall_time"] for m in ntn_metrics]
ntn_vals = [gtn_init_val] + [m["val_quality"]*100 for m in ntn_metrics]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(ntn_times, ntn_vals, "o-", color="#1f77b4", label="N-CMPO2 (NTN)", markersize=4, linewidth=1.5)
ax.plot(gtn_times, gtn_vals, "s-", color="#ff7f0e", label="G-CMPO2 (GTN)", markersize=3, linewidth=1.5, alpha=0.8)

ax.set_xlabel("Wall Time (seconds)", fontsize=12)
ax.set_ylabel("Validation Quality (%)", fontsize=12)
ax.set_title("N-CMPO2 vs G-CMPO2: Validation Quality vs Wall Time\n(MNIST)", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {OUTPUT_PATH}")
plt.close()
