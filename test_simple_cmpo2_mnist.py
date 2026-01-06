# type: ignore
"""
Test SimpleCMPO2_NTN (no caching) on MNIST classification
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
import quimb.tensor as qt
from model.MPS_simple import SimpleCMPO2_NTN
from model.builder import Inputs
from model.losses import CrossEntropyLoss
from model.utils import CLASSIFICATION_METRICS

torch.set_default_dtype(torch.float32)

print("="*70)
print("SIMPLE CMPO2_NTN ON MNIST (NO CACHING)")
print("="*70)

# Load MNIST
print("\n1. Loading MNIST data...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, 
                                          transform=transform, download=True)

# Use smaller subset
SUBSET_SIZE = 3000  # Much smaller for testing
BATCH_SIZE = 100
N_EPOCHS = 2

train_loader_raw = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Collect training data
train_samples = []
train_labels = []
for images, labels in train_loader_raw:
    train_samples.append(images)
    train_labels.append(labels)
    if len(torch.cat(train_samples)) >= SUBSET_SIZE:
        break

x_data = torch.cat(train_samples, dim=0)[:SUBSET_SIZE]
y_data = torch.cat(train_labels, dim=0)[:SUBSET_SIZE]

# Unfold images into patches (4x4 patches with stride 4)
KERNEL_SIZE = 4
STRIDE = 4

x_data = F.unfold(x_data, kernel_size=(KERNEL_SIZE, KERNEL_SIZE), 
                     stride=(STRIDE, STRIDE), padding=0).transpose(-2, -1)

# Add bias patch and bias pixel
x_data = torch.cat((x_data, torch.zeros((x_data.shape[0], 1, x_data.shape[2]))), dim=1)
x_data = torch.cat((x_data, torch.zeros((x_data.shape[0], x_data.shape[1], 1))), dim=2)
x_data[..., -1, -1] = 1.0

DIM_PATCHES = x_data.shape[1]  # 50
DIM_PIXELS = x_data.shape[2]   # 17
N_CLASSES = 10
L = 3
BOND_DIM = 3

print(f"\nConfiguration:")
print(f"  Sites (L): {L}")
print(f"  Bond dimension: {BOND_DIM}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Total samples: {SUBSET_SIZE}")
print(f"  Training epochs: {N_EPOCHS}")
print(f"  Number of classes: {N_CLASSES}")
print(f"  Patches: {DIM_PATCHES}, Pixels: {DIM_PIXELS}")

print(f"\nData:")
print(f"  X shape: {x_data.shape}")
print(f"  y shape: {y_data.shape}")
print(f"  Class distribution: {torch.bincount(y_data)}")

# Setup MPS with output dimension = N_CLASSES
psi = qt.MPS_rand_state(L, bond_dim=BOND_DIM, phys_dim=DIM_PIXELS)
phi = qt.MPS_rand_state(L, bond_dim=BOND_DIM, phys_dim=DIM_PATCHES)

# Add output dimension to middle node
middle_psi = psi['I1']
middle_psi.new_ind('out', size=N_CLASSES, axis=-1, mode='random', rand_strength=0.01)
psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))
phi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))

# Reindex physical dimensions
psi.reindex({f"k{i}": f"{i}_pixels" for i in range(L)}, inplace=True)
phi.reindex({f"k{i}": f"{i}_patches" for i in range(L)}, inplace=True)

# Add tags
for i in range(L):
    psi.add_tag(f"{i}_Pi", where=f"I{i}")
    phi.add_tag(f"{i}_Pa", where=f"I{i}")

tn = psi & phi

print(f"\nTensor Network:")
print(f"  Total tensors: {len(tn.tensors)}")
print(f"  Outer indices: {tn.outer_inds()}")

# Setup data loader
# Input labels format: [source_idx, (patch_ind, pixel_ind)]
# All three sites reference the same source data (index 0)
input_labels_cmpo2 = [
    [0, (f"{i}_patches", f"{i}_pixels")]
    for i in range(L)
]

# Simple labels for model.input_dims
input_labels = [str(i) for i in range(L)]

loader = Inputs(
    inputs=[x_data],
    outputs=[y_data.unsqueeze(1)],  # NTN expects (N, 1) for class indices
    outputs_labels=["out"],
    input_labels=input_labels_cmpo2,
    batch_dim="s",
    batch_size=BATCH_SIZE
)

# Setup loss
loss = CrossEntropyLoss()

# Create simple model (NO CACHING)
model = SimpleCMPO2_NTN(
    tn=tn,
    output_dims=["out"],
    input_dims=input_labels,
    loss=loss,
    data_stream=loader,
    psi=psi,
    phi=phi
)

print(f"\nModel:")
print(f"  Type: SimpleCMPO2_NTN (no caching)")
print(f"  Loss: CrossEntropyLoss")

print(f"\n" + "-"*70)
print("TRAINING")
print("-"*70)

# Jitter schedule: keep at 10 for first 3 epochs, then decay by 0.5 with minimum 1e-3
jitter_schedule = []
for epoch in range(N_EPOCHS):
    jitter_val = 1.0 * (0.1 ** (epoch ))
    jitter_schedule.append(max(jitter_val, 1e-8))  # Minimum 1e-3

print(f"Jitter schedule: {[f'{j:.4f}' for j in jitter_schedule]}")

try:
    final_metrics = model.fit(
        n_epochs=N_EPOCHS,
        regularize=True,
        jitter=jitter_schedule,
        eval_metrics=CLASSIFICATION_METRICS,
        verbose=True
    )
    
    print(f"\n" + "-"*70)
    print("TRAINING COMPLETE")
    print("-"*70)
    
    # Print final metrics
    print(f"\nFinal Metrics (Epoch {N_EPOCHS}):")
    print(f"  Loss: {final_metrics['loss']:.4f}")
    print(f"  Accuracy: {final_metrics['accuracy']:.2%}")
    
    # Check if loss is valid
    import math
    if math.isnan(final_metrics['loss']):
        print(f"\n✗ ERROR: Loss is NaN - training unstable!")
    elif final_metrics['accuracy'] > 0.15:  # Random baseline for 10 classes is 10%
        print(f"\n✓ SUCCESS: Accuracy {final_metrics['accuracy']:.2%} is above random baseline (10%)")
    else:
        print(f"\n✗ WARNING: Accuracy {final_metrics['accuracy']:.2%} is close to random baseline (10%)")

except Exception as e:
    print(f"\n✗ ERROR during training:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
