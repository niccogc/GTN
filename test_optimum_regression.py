# type: ignore
"""Compare fit() with use_lstsq=True vs False on abalone dataset."""

import torch
import time
import tracemalloc
import gc

torch.set_default_dtype(torch.float64)

from model.base.NTN import NTN
from model.losses import MSELoss
from model.utils import create_inputs, REGRESSION_METRICS
from model.standard import MPO2
from experiments.dataset_loader import load_dataset

print("Loading abalone dataset (regression)...")
data, info = load_dataset("abalone")
print(f"Train: {data['X_train'].shape}, Val: {data['X_val'].shape}")

L = 3
bond_dim = 12
input_dim = data["X_train"].shape[1]
N_EPOCHS = 3
JITTER = 10

print(f"Config: L={L}, bond_dim={bond_dim}, n_epochs={N_EPOCHS}, jitter={JITTER}")


def create_ntn():
    model = MPO2(L=L, bond_dim=bond_dim, phys_dim=input_dim, output_dim=1)
    loader_train = create_inputs(
        X=data["X_train"],
        y=data["y_train"],
        input_labels=model.input_labels,
        output_labels=model.output_dims,
        batch_size=1024,
        append_bias=False,
    )
    loader_val = create_inputs(
        X=data["X_val"],
        y=data["y_val"],
        input_labels=model.input_labels,
        output_labels=model.output_dims,
        batch_size=64,
        append_bias=False,
    )
    ntn = NTN(
        tn=model.tn,
        output_dims=model.output_dims,
        input_dims=model.input_dims,
        loss=MSELoss(),
        data_stream=loader_train,
    )
    return ntn, loader_val


# --- NEWTON METHOD ---
print("\n" + "=" * 70)
print("NEWTON METHOD (use_lstsq=False)")
print("=" * 70)

gc.collect()
torch.manual_seed(42)
ntn_newton, val_newton = create_ntn()

tracemalloc.start()
start = time.perf_counter()
scores_train_n, scores_val_n = ntn_newton.fit(
    n_epochs=N_EPOCHS,
    jitter=JITTER,
    val_data=val_newton,
    eval_metrics=REGRESSION_METRICS,
    use_lstsq=False,
    verbose=True,
)
time_newton = time.perf_counter() - start
mem_newton_current, mem_newton_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

# --- LSTSQ METHOD ---
print("\n" + "=" * 70)
print("LSTSQ METHOD (use_lstsq=True)")
print("=" * 70)

gc.collect()
torch.manual_seed(42)
ntn_lstsq, val_lstsq = create_ntn()

tracemalloc.start()
start = time.perf_counter()
scores_train_l, scores_val_l = ntn_lstsq.fit(
    n_epochs=N_EPOCHS,
    jitter=JITTER,
    val_data=val_lstsq,
    eval_metrics=REGRESSION_METRICS,
    use_lstsq=True,
    verbose=True,
)
time_lstsq = time.perf_counter() - start
mem_lstsq_current, mem_lstsq_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

# --- SUMMARY ---
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Metric':<20} {'Newton':<15} {'LSTSQ':<15} {'Winner'}")
print("-" * 60)

# Loss comparison
newton_loss = scores_val_n['loss']
lstsq_loss = scores_val_l['loss']
winner = "LSTSQ" if lstsq_loss < newton_loss else "Newton"
print(f"{'Val Loss':<20} {newton_loss:<15.4f} {lstsq_loss:<15.4f} {winner}")

# Time comparison
winner = "LSTSQ" if time_lstsq < time_newton else "Newton"
print(f"{'Time (s)':<20} {time_newton:<15.2f} {time_lstsq:<15.2f} {winner}")

# Memory comparison
mem_newton_mb = mem_newton_peak / 1024 / 1024
mem_lstsq_mb = mem_lstsq_peak / 1024 / 1024
winner = "LSTSQ" if mem_lstsq_mb < mem_newton_mb else "Newton"
print(f"{'Peak Memory (MB)':<20} {mem_newton_mb:<15.2f} {mem_lstsq_mb:<15.2f} {winner}")

print()
print(f"Speedup: {time_newton/time_lstsq:.2f}x")
print(f"Memory ratio: {mem_newton_peak/mem_lstsq_peak:.2f}x")
