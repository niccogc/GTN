# type: ignore
"""
Basic test for CMPO2_NTN to verify it works before tackling MNIST.
"""
import torch
import quimb.tensor as qt
from model.MPS import CMPO2_NTN
from model.builder import Inputs
from model.losses import MSELoss

print("="*60)
print("Basic CMPO2_NTN Test")
print("="*60)

# Simple parameters - mimicking MNIST structure
BATCH_SIZE = 10
N_SAMPLES = 50
DIM_PATCHES = 5
DIM_PIXELS = 4
BOND_DIM = 2
N_OUTPUTS = 2

print(f"\nSetup:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Samples: {N_SAMPLES}")
print(f"  Dimensions: patches={DIM_PATCHES}, pixels={DIM_PIXELS}")
print(f"  Bond dim: {BOND_DIM}")

# Create MPS objects EXACTLY like in test_environment_caching.py
L = 3  # 3 sites

# Create two MPS
psi = qt.MPS_rand_state(L, bond_dim=BOND_DIM, phys_dim=DIM_PIXELS)
phi = qt.MPS_rand_state(L, bond_dim=BOND_DIM, phys_dim=DIM_PATCHES)

# Add output index to middle psi tensor (site 1)
middle_psi = psi['I1']
# Use new_ind to add 'out' dimension properly
middle_psi.new_ind('out', size=N_OUTPUTS, axis=-1, mode='random', rand_strength=0.1)
# Convert to torch
psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))
phi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))

# Reindex to match our naming
psi.reindex({f"k{i}": f"{i}_pixels" for i in range(L)}, inplace=True)
phi.reindex({f"k{i}": f"{i}_patches" for i in range(L)}, inplace=True)

# Add unique block-level name tags
for i in range(L):
    psi.add_tag(f"{i}_Pi", where=f"I{i}")
    phi.add_tag(f"{i}_Pa", where=f"I{i}")


print("\n✓ Created MPS objects")
print(f"  Pixel-MPS (psi): {L} sites")
print(f"  Patch-MPS (phi): {L} sites")
print(f"  Added 'out' index to psi[I1]")

# Combine the two MPS
tn = psi & phi

print(f"✓ Combined MPS into TensorNetwork1D")
print(f"  TN type: {type(tn)}")
print(f"  TN has .L: {hasattr(tn, 'L')}, value: {tn.L if hasattr(tn, 'L') else 'N/A'}")

# Create data with shape (samples, patches, pixels) like MNIST
x_data = torch.randn(N_SAMPLES, DIM_PATCHES, DIM_PIXELS)
y_data = torch.randn(N_SAMPLES, N_OUTPUTS)

print(f"\n✓ Created data")
print(f"  Input shape: {x_data.shape}")
print(f"  Output shape: {y_data.shape}")

# Setup data loader with correct format
# input_labels_ntn points all sites to same source with different index pairs
input_labels_ntn = [
    [0, ("0_patches", "0_pixels")],
    [0, ("1_patches", "1_pixels")],
    [0, ("2_patches", "2_pixels")]
]
input_dims_simple = ["0", "1", "2"]

loader = Inputs(
    inputs=[x_data],  # Single source
    outputs=[y_data],
    outputs_labels=["out"],
    input_labels=input_labels_ntn,
    batch_dim="s",
    batch_size=BATCH_SIZE
)

print("✓ Created data loader")

# Create model - pass the TN directly!
loss = MSELoss()
model = CMPO2_NTN(
    tn=tn,
    output_dims=["out"],
    input_dims=input_dims_simple,
    loss=loss,
    data_stream=loader,
    cache_environments=True  # TODO: Fix structure for caching
)

print("✓ Created CMPO2_NTN model")
print(f"  Trainable nodes: {model._get_trainable_nodes()}")

# Test one epoch of training
print("\nTesting one epoch of training with caching...")
try:
    metrics = model.fit(n_epochs=1, regularize=True, jitter=1e-4, verbose=True)
    print(f"✓ Training successful!")
    print(f"  Final metrics: {metrics}")
    
    # Print cache statistics
    cache_stats = model.get_cache_stats()
    print(f"\n✓ Cache Statistics:")
    print(f"  Hits: {cache_stats['hits']}")
    print(f"  Misses: {cache_stats['misses']}")
    print(f"  Total calls: {cache_stats['total']}")
    print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")
except Exception as e:
    print(f"✗ Training failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Basic test complete!")
print("="*60)
