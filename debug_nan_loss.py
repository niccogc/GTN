# type: ignore
"""
Debug why loss becomes NaN - copy exactly from NTN code
"""
import torch
import quimb.tensor as qt
from model.MPS_simple import SimpleCMPO2_NTN
from model.builder import Inputs
from model.losses import CrossEntropyLoss
from model.utils import metric_cross_entropy, metric_accuracy

# Simple setup - exactly like test
L = 3
BOND_DIM = 3
DIM_PATCHES = 5
DIM_PIXELS = 4
N_CLASSES = 3

psi = qt.MPS_rand_state(L, bond_dim=BOND_DIM, phys_dim=DIM_PIXELS)
phi = qt.MPS_rand_state(L, bond_dim=BOND_DIM, phys_dim=DIM_PATCHES)

middle_psi = psi['I1']
middle_psi.new_ind('out', size=N_CLASSES, axis=-1, mode='random', rand_strength=0.01)
psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))
phi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))

psi.reindex({f"k{i}": f"{i}_pixels" for i in range(L)}, inplace=True)
phi.reindex({f"k{i}": f"{i}_patches" for i in range(L)}, inplace=True)

for i in range(L):
    psi.add_tag(f"{i}_Pi", where=f"I{i}")
    phi.add_tag(f"{i}_Pa", where=f"I{i}")

tn = psi & phi

# Create data
N_SAMPLES = 20
x_data = torch.randn(N_SAMPLES, DIM_PATCHES, DIM_PIXELS)
y_data = torch.randint(0, N_CLASSES, (N_SAMPLES,))

input_labels = [[0, (f"{i}_patches", f"{i}_pixels")] for i in range(L)]

loader = Inputs(
    inputs=[x_data],
    outputs=[y_data.unsqueeze(1)],
    outputs_labels=["out"],
    input_labels=input_labels,
    batch_dim="s",
    batch_size=5
)

loss_fn = CrossEntropyLoss()
model = SimpleCMPO2_NTN(
    tn=tn,
    output_dims=["out"],
    input_dims=[str(i) for i in range(L)],
    loss=loss_fn,
    data_stream=loader
)

print("="*70)
print("DEBUGGING NaN LOSS")
print("="*70)

# Do EXACTLY what NTN does in _batch_evaluate
print("\n1. Initial evaluation (before any updates)...")
batch = next(iter(loader))
inputs, y_true = batch

with torch.no_grad():
    output_inds = [model.batch_dim] + model.output_dimensions
    y_pred = model._batch_forward(inputs, model.tn, output_inds)
    
    y_pred_th = model._to_torch(y_pred)
    y_true_th = model._to_torch(y_true)
    
    print(f"   y_pred_th shape: {y_pred_th.shape}, dtype: {y_pred_th.dtype}")
    print(f"   y_true_th shape: {y_true_th.shape}, dtype: {y_true_th.dtype}")
    print(f"   y_pred_th stats: min={y_pred_th.min():.4f}, max={y_pred_th.max():.4f}, mean={y_pred_th.mean():.4f}")
    
    # Call metric exactly as NTN does
    loss_val, count = metric_cross_entropy(y_pred_th, y_true_th)
    acc_val, count2 = metric_accuracy(y_pred_th, y_true_th)
    
    print(f"   Initial loss: {loss_val:.4f}, count: {count}")
    print(f"   Initial accuracy: {acc_val}/{count2} = {acc_val/count2:.2%}")
    print(f"   Loss is NaN: {torch.isnan(loss_val)}")

# Do one update with jitter=10
print("\n2. Update first node with jitter=10...")
node = '0_Pi'
print(f"   Updating node: {node}")
model.update_tn_node(node, regularize=True, jitter=10.0)

# Evaluate again
print("\n3. Evaluation after update...")
with torch.no_grad():
    y_pred = model._batch_forward(inputs, model.tn, output_inds)
    
    y_pred_th = model._to_torch(y_pred)
    
    print(f"   y_pred_th shape: {y_pred_th.shape}")
    print(f"   y_pred_th stats: min={y_pred_th.min():.4f}, max={y_pred_th.max():.4f}, mean={y_pred_th.mean():.4f}")
    print(f"   Has NaN: {torch.isnan(y_pred_th).any()}")
    print(f"   Has Inf: {torch.isinf(y_pred_th).any()}")
    
    loss_val, count = metric_cross_entropy(y_pred_th, y_true_th)
    acc_val, count2 = metric_accuracy(y_pred_th, y_true_th)
    
    print(f"   Loss after update: {loss_val:.4f}")
    print(f"   Accuracy: {acc_val}/{count2} = {acc_val/count2:.2%}")
    print(f"   Loss is NaN: {torch.isnan(loss_val)}")

# Do a full sweep
print("\n4. Doing full sweep (all nodes)...")
nodes = model._get_trainable_nodes()
print(f"   Nodes to update: {nodes}")

for i, node in enumerate(nodes):
    print(f"\n   Updating {i+1}/{len(nodes)}: {node}...")
    model.update_tn_node(node, regularize=True, jitter=10.0)
    
    # Check after each update
    with torch.no_grad():
        y_pred = model._batch_forward(inputs, model.tn, output_inds)
        y_pred_th = model._to_torch(y_pred)
        
        has_nan = torch.isnan(y_pred_th).any()
        has_inf = torch.isinf(y_pred_th).any()
        
        if has_nan or has_inf:
            print(f"   ERROR: NaN/Inf detected after updating {node}!")
            print(f"     Has NaN: {has_nan}")
            print(f"     Has Inf: {has_inf}")
            break
        
        loss_val, _ = metric_cross_entropy(y_pred_th, y_true_th)
        acc_val, count = metric_accuracy(y_pred_th, y_true_th)
        
        print(f"     Loss: {loss_val:.4f}, Accuracy: {acc_val/count:.2%}")
        
        if torch.isnan(loss_val):
            print(f"   ERROR: Loss is NaN after updating {node}!")
            break

# Evaluate on ALL batches
print("\n5. Evaluating on ALL batches...")
total_loss = 0
total_count = 0
total_correct = 0

for batch_idx, batch_data in enumerate(loader.data_mu_y):
    batch_inputs, batch_y_true = batch_data
    
    with torch.no_grad():
        batch_y_pred = model._batch_forward(batch_inputs, model.tn, output_inds)
        batch_y_pred_th = model._to_torch(batch_y_pred)
        batch_y_true_th = model._to_torch(batch_y_true)
        
        batch_loss, batch_count = metric_cross_entropy(batch_y_pred_th, batch_y_true_th)
        batch_correct, _ = metric_accuracy(batch_y_pred_th, batch_y_true_th)
        
        total_loss += batch_loss
        total_count += batch_count
        total_correct += batch_correct
        
        print(f"   Batch {batch_idx}: loss={batch_loss:.4f}, acc={batch_correct/batch_count:.2%}, has_nan={torch.isnan(batch_loss)}")
        
        if torch.isnan(batch_loss):
            print(f"   ERROR: Batch {batch_idx} has NaN loss!")
            print(f"   y_pred stats: min={batch_y_pred_th.min():.4f}, max={batch_y_pred_th.max():.4f}")
            break

avg_loss = (total_loss / total_count).item()
avg_acc = (total_correct / total_count).item()
print(f"\n   Overall: loss={avg_loss:.4f}, accuracy={avg_acc:.2%}")
print(f"   Overall loss is NaN: {torch.isnan(torch.tensor(avg_loss))}")

print("\n" + "="*70)
