# type: ignore
"""
Debug exactly when NaN appears during training
"""
import torch
import quimb.tensor as qt
from model.MPS_simple import SimpleCMPO2_NTN
from model.builder import Inputs
from model.losses import CrossEntropyLoss
from model.utils import metric_cross_entropy, metric_accuracy

print("="*70)
print("DEBUGGING NaN DURING SWEEP")
print("="*70)

# Setup
L = 2
BOND_DIM = 8
DIM_PATCHES = 5
DIM_PIXELS = 4
N_CLASSES = 3
N_SAMPLES = 100
BATCH_SIZE = 10

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

# Generate data
x_data = []
y_data = []
for i in range(N_SAMPLES):
    class_id = i % N_CLASSES
    if class_id == 0:
        x = torch.randn(DIM_PATCHES, DIM_PIXELS) + 1.0
    elif class_id == 1:
        x = torch.randn(DIM_PATCHES, DIM_PIXELS) - 1.0
    else:
        x = torch.randn(DIM_PATCHES, DIM_PIXELS) * 0.5
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
    batch_size=BATCH_SIZE
)

loss_fn = CrossEntropyLoss()
model = SimpleCMPO2_NTN(
    tn=tn,
    output_dims=["out"],
    input_dims=[str(i) for i in range(L)],
    loss=loss_fn,
    data_stream=loader,
    psi=psi,
    phi=phi
)

print(f"\nInitial node norms:")
for tag in model._get_trainable_nodes():
    norm = torch.norm(model.tn[tag].data).item()
    print(f"  {tag}: {norm:.4f}")

# Do 3 epochs with detailed logging
trainable_nodes = model._get_trainable_nodes()
back_sweep = trainable_nodes[-2:0:-1]
full_sweep_order = trainable_nodes + back_sweep

print(f"\nSweep order: {full_sweep_order}")

for epoch in range(3):
    print(f"\n{'='*70}")
    print(f"EPOCH {epoch + 1}")
    print(f"{'='*70}")
    
    for node_idx, node_tag in enumerate(full_sweep_order):
        print(f"\n  Updating node {node_idx+1}/{len(full_sweep_order)}: {node_tag}")
        
        # Check node norm before update
        node_before = model.tn[node_tag].data
        norm_before = torch.norm(node_before).item()
        print(f"    Before: norm={norm_before:.4f}, min={node_before.min():.4f}, max={node_before.max():.4f}")
        
        # Update
        model.update_tn_node(node_tag, regularize=True, jitter=10.0)
        
        # Check node norm after update
        node_after = model.tn[node_tag].data
        norm_after = torch.norm(node_after).item()
        has_nan = torch.isnan(node_after).any().item()
        has_inf = torch.isinf(node_after).any().item()
        print(f"    After:  norm={norm_after:.4f}, min={node_after.min():.4f}, max={node_after.max():.4f}")
        print(f"    Has NaN: {has_nan}, Has Inf: {has_inf}")
        
        if has_nan or has_inf:
            print(f"\n  ⚠️ NaN/Inf DETECTED in {node_tag} after update!")
            print(f"    This is the problem node!")
            exit(1)
        
        # Normalize (manual version)
        pi_tensors = [model.tn[tag] for tag in model._get_trainable_nodes() if '_Pi' in tag]
        if pi_tensors:
            total_norm_sq = sum(torch.sum(t.data ** 2).item() for t in pi_tensors)
            norm = total_norm_sq ** 0.5
            if norm > 0:
                for t in pi_tensors:
                    t.modify(data=t.data / norm)
        
        pa_tensors = [model.tn[tag] for tag in model._get_trainable_nodes() if '_Pa' in tag]
        if pa_tensors:
            total_norm_sq = sum(torch.sum(t.data ** 2).item() for t in pa_tensors)
            norm = total_norm_sq ** 0.5
            if norm > 0:
                for t in pa_tensors:
                    t.modify(data=t.data / norm)
        
        # Check all nodes after normalization
        print(f"    After normalize - all norms:")
        for tag in model._get_trainable_nodes():
            norm = torch.norm(model.tn[tag].data).item()
            print(f"      {tag}: {norm:.4f}")
    
    # Evaluate
    print(f"\n  Evaluating...")
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
            
            if torch.isnan(batch_loss):
                print(f"    ⚠️ NaN loss in batch!")
                print(f"    y_pred stats: min={batch_y_pred_th.min():.4f}, max={batch_y_pred_th.max():.4f}")
                print(f"    Has NaN in predictions: {torch.isnan(batch_y_pred_th).any()}")
                break
    
    avg_loss = (total_loss / total_count).item()
    avg_acc = (total_correct / total_count).item()
    
    print(f"\n  Epoch {epoch+1} results:")
    print(f"    Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2%}")
    print(f"    Loss is NaN: {torch.isnan(torch.tensor(avg_loss))}")
    
    if torch.isnan(torch.tensor(avg_loss)):
        print(f"\n  ⚠️ NaN loss detected after epoch {epoch+1}!")
        break

print("\n" + "="*70)
