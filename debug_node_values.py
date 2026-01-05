# type: ignore
"""
Debug node values during training to find explosion source
"""
import torch
import quimb.tensor as qt
from model.MPS_simple import SimpleCMPO2_NTN
from model.builder import Inputs
from model.losses import CrossEntropyLoss
from model.utils import CLASSIFICATION_METRICS

L = 2
BOND_DIM = 3
DIM_PATCHES = 5
DIM_PIXELS = 4
N_CLASSES = 10
N_SAMPLES = 100

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
x_data = []
y_data = []
for i in range(N_SAMPLES):
    class_id = i % N_CLASSES
    x = torch.randn(DIM_PATCHES, DIM_PIXELS) 
    x_data.append(x)
    y_data.append(class_id)

x_data = torch.stack(x_data)
y_data = torch.tensor(y_data, dtype=torch.long)

input_labels = [[0, (f"{i}_patches", f"{i}_pixels")] for i in range(L)]

loader = Inputs(
    inputs=[x_data],
    outputs=[y_data.unsqueeze(1)],
    outputs_labels=["out"],
    input_labels=input_labels,
    batch_dim="s",
    batch_size=10
)

loss = CrossEntropyLoss()
model = SimpleCMPO2_NTN(
    tn=tn,
    output_dims=["out"],
    input_dims=[str(i) for i in range(L)],
    loss=loss,
    data_stream=loader
)

print("="*70)
print("DEBUGGING NODE VALUES")
print("="*70)

# Print initial node statistics
print("\nInitial node statistics:")
for node_tag in model._get_trainable_nodes():
    node_tensor = model.tn[node_tag].data
    print(f"  {node_tag}: shape={node_tensor.shape}, norm={torch.norm(node_tensor):.4f}, "
          f"min={node_tensor.min():.4f}, max={node_tensor.max():.4f}")

# Do ONE update and check
print("\n" + "="*70)
print("Updating first node (0_Pi) with jitter=1e-4...")
print("="*70)

node_tag = '0_Pi'
node_before = model.tn[node_tag].data.clone()

model.update_tn_node(node_tag, regularize=True, jitter=1e-4)

node_after = model.tn[node_tag].data

print(f"\nNode '{node_tag}' before update:")
print(f"  norm: {torch.norm(node_before):.4f}")
print(f"  min: {node_before.min():.4f}, max: {node_before.max():.4f}")
print(f"  mean: {node_before.mean():.4f}, std: {node_before.std():.4f}")

print(f"\nNode '{node_tag}' after update:")
print(f"  norm: {torch.norm(node_after):.4f}")
print(f"  min: {node_after.min():.4f}, max: {node_after.max():.4f}")
print(f"  mean: {node_after.mean():.4f}, std: {node_after.std():.4f}")
print(f"  has NaN: {torch.isnan(node_after).any()}")
print(f"  has Inf: {torch.isinf(node_after).any()}")

norm_ratio = torch.norm(node_after) / torch.norm(node_before)
print(f"\nNorm ratio (after/before): {norm_ratio:.4f}")

if norm_ratio > 100:
    print("  ⚠️ WARNING: Node norm exploded by >100x!")
elif norm_ratio > 10:
    print("  ⚠️ WARNING: Node norm increased by >10x!")
elif norm_ratio < 0.01:
    print("  ⚠️ WARNING: Node norm collapsed by >100x!")

# Do a full sweep
print("\n" + "="*70)
print("Doing full sweep...")
print("="*70)

nodes = model._get_trainable_nodes()
for i, node_tag in enumerate(nodes):
    node_before = model.tn[node_tag].data.clone()
    
    model.update_tn_node(node_tag, regularize=True, jitter=1e-4)
    
    node_after = model.tn[node_tag].data
    norm_ratio = torch.norm(node_after) / torch.norm(node_before)
    
    print(f"{i+1}. {node_tag}: norm {torch.norm(node_before):.4f} -> {torch.norm(node_after):.4f} (ratio={norm_ratio:.2f})")
    
    if torch.isnan(node_after).any() or torch.isinf(node_after).any():
        print(f"   ⚠️ ERROR: NaN/Inf detected in {node_tag}!")
        break

# Check all nodes after sweep
print("\nAll node statistics after sweep:")
for node_tag in model._get_trainable_nodes():
    node_tensor = model.tn[node_tag].data
    print(f"  {node_tag}: norm={torch.norm(node_tensor):.4f}, "
          f"min={node_tensor.min():.4f}, max={node_tensor.max():.4f}")

# Continue with multiple epochs
print("\n" + "="*70)
print("Running multiple epochs...")
print("="*70)

from model.utils import metric_cross_entropy, metric_accuracy

for epoch in range(7):
    print(f"\nEpoch {epoch+1}:")
    
    # Full sweep
    nodes = model._get_trainable_nodes()
    back_sweep = nodes[-2:0:-1]
    full_sweep_order = nodes + back_sweep
    
    for node_tag in full_sweep_order:
        model.update_tn_node(node_tag, regularize=True, jitter=1e-4)
    
    # Evaluate
    total_loss = 0
    total_count = 0
    total_correct = 0
    
    for batch_inputs, batch_y_true in loader.data_mu_y:
        with torch.no_grad():
            output_inds = [model.batch_dim] + model.output_dimensions
            batch_y_pred = model._batch_forward(batch_inputs, model.tn, output_inds)
            batch_y_pred_th = model._to_torch(batch_y_pred)
            batch_y_true_th = model._to_torch(batch_y_true)
            
            batch_loss, batch_count = metric_cross_entropy(batch_y_pred_th, batch_y_true_th)
            batch_correct, _ = metric_accuracy(batch_y_pred_th, batch_y_true_th)
            
            total_loss += batch_loss
            total_count += batch_count
            total_correct += batch_correct
    
    avg_loss = (total_loss / total_count).item()
    avg_acc = (total_correct / total_count).item()
    
    print(f"  Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2%}")
    print(f"  Loss is NaN: {torch.isnan(torch.tensor(avg_loss))}")
    
    # Check node norms
    print(f"  Node norms: ", end="")
    for node_tag in model._get_trainable_nodes():
        norm = torch.norm(model.tn[node_tag].data).item()
        print(f"{node_tag}={norm:.2f} ", end="")
    print()
    
    # Check for explosion
    max_norm = max(torch.norm(model.tn[node_tag].data).item() for node_tag in model._get_trainable_nodes())
    if max_norm > 1000:
        print(f"  ⚠️ EXPLOSION: Max norm = {max_norm:.2f}")
        break
    if torch.isnan(torch.tensor(avg_loss)):
        print(f"  ⚠️ NaN loss detected!")
        break

print("\n" + "="*70)
