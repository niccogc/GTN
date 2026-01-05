# type: ignore
"""
Test SimpleCMPO2_NTN (no caching) for classification
"""
import torch
import quimb.tensor as qt
from model.MPS_simple import SimpleCMPO2_NTN
from model.builder import Inputs
from model.losses import CrossEntropyLoss
from model.utils import CLASSIFICATION_METRICS

print("="*70)
print("SIMPLE CMPO2_NTN CLASSIFICATION TEST (NO CACHING)")
print("="*70)

BATCH_SIZE = 10
N_SAMPLES = 100
DIM_PATCHES = 5
DIM_PIXELS = 4
BOND_DIM = 8  # Increased from 3 to 8
N_CLASSES = 9  # Start with 3 classes
N_EPOCHS = 20  # More epochs
L = 3  # Use 3 sites

print(f"\nConfiguration:")
print(f"  Sites (L): {L}")
print(f"  Bond dimension: {BOND_DIM}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Total samples: {N_SAMPLES}")
print(f"  Training epochs: {N_EPOCHS}")
print(f"  Number of classes: {N_CLASSES}")

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

# Generate synthetic classification data
# Create 3 distinct patterns for 3 classes
x_data = []
y_data = []

for i in range(N_SAMPLES):
    class_id = i % N_CLASSES
    
    # Create somewhat distinct patterns per class
    if class_id == 0:
        x = torch.randn(DIM_PATCHES, DIM_PIXELS) + 1.0  # Shifted positive
    elif class_id == 1:
        x = torch.randn(DIM_PATCHES, DIM_PIXELS) - 1.0  # Shifted negative
    else:
        x = torch.randn(DIM_PATCHES, DIM_PIXELS) * 0.5  # Smaller variance
    
    x_data.append(x)
    y_data.append(class_id)

x_data = torch.stack(x_data)  # Shape: [N_SAMPLES, DIM_PATCHES, DIM_PIXELS]
y_data = torch.tensor(y_data, dtype=torch.long)  # Shape: [N_SAMPLES]

print(f"\nData:")
print(f"  X shape: {x_data.shape}")
print(f"  y shape: {y_data.shape}")
print(f"  Class distribution: {torch.bincount(y_data)}")

# Setup data loader
input_labels_ntn = [
    [0, (f"{i}_patches", f"{i}_pixels")]
    for i in range(L)
]

loader = Inputs(
    inputs=[x_data],
    outputs=[y_data.unsqueeze(1)],  # NTN expects (N, 1) for class indices
    outputs_labels=["out"],
    input_labels=input_labels_ntn,
    batch_dim="s",
    batch_size=BATCH_SIZE
)

# Setup loss
loss = CrossEntropyLoss()

# Create simple model (NO CACHING)
model = SimpleCMPO2_NTN(
    tn=tn,
    output_dims=["out"],
    input_dims=[str(i) for i in range(L)],
    loss=loss,
    data_stream=loader,
    psi=psi,  # Pass the MPS objects
    phi=phi
)

print(f"\nModel:")
print(f"  Type: SimpleCMPO2_NTN (no caching)")
print(f"  Loss: CrossEntropyLoss")

print(f"\n" + "-"*70)
print("TRAINING")
print("-"*70)

try:
    # Keep jitter at 10 for first 3 epochs, then decay much faster by 0.5 with minimum 0.1
    jitter_schedule = []
    for epoch in range(N_EPOCHS):
        if epoch < 3:
            jitter_schedule.append(10.0)
        else:
            jitter_val = 10.0 * (0.5 ** (epoch - 3))
            jitter_schedule.append(max(jitter_val, 0.01))  # Minimum 0.1
    
    print(f"Jitter schedule: {[f'{j:.4f}' for j in jitter_schedule]}")
    
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
        print(f"  This likely means gradients exploded or became invalid")
        print(f"  Try: smaller jitter, lower learning rate, or different initialization")
    elif final_metrics['accuracy'] > 0.4:
        print(f"\n✓ SUCCESS: Accuracy {final_metrics['accuracy']:.2%} is above random baseline (33.3%)")
    else:
        print(f"\n✗ WARNING: Accuracy {final_metrics['accuracy']:.2%} is close to random baseline (33.3%)")
    
    # Test prediction on a few samples (if loss is valid)
    if not math.isnan(final_metrics['loss']):
        print(f"\n" + "-"*70)
        print("SAMPLE PREDICTIONS")
        print("-"*70)
        
        test_batch = next(iter(loader))
        test_inputs, test_labels = test_batch
        
        # Forward pass
        logits = model.forward(test_inputs)  # Shape: [batch, n_classes]
        predictions = torch.argmax(logits.data, dim=-1)
        
        print(f"\nFirst 10 predictions vs true labels:")
        for i in range(min(10, len(predictions))):
            pred = predictions[i].item()
            true = test_labels.data[i, 0].item()  # test_labels is [batch, 1]
            match = "✓" if pred == true else "✗"
            print(f"  Sample {i}: pred={pred}, true={true} {match}")

except Exception as e:
    print(f"\n✗ ERROR during training:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
