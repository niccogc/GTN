# type: ignore
"""
Test CMPO2_NTN for classification with accuracy metrics
"""
import torch
import torch.nn.functional as F
import quimb.tensor as qt
from model.MPS import CMPO2_NTN
from model.builder import Inputs
from model.losses import CrossEntropyLoss
from model.utils import CLASSIFICATION_METRICS

print("="*70)
print("CMPO2_NTN CLASSIFICATION TEST")
print("="*70)

BATCH_SIZE = 10
N_SAMPLES = 100
DIM_PATCHES = 5
DIM_PIXELS = 4
BOND_DIM = 3
N_CLASSES = 10  # 10-class classification like MNIST
N_EPOCHS = 5

print(f"\nConfiguration:")
print(f"  Sites (L): 3")
print(f"  Bond dimension: {BOND_DIM}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Total samples: {N_SAMPLES}")
print(f"  Training epochs: {N_EPOCHS}")
print(f"  Number of classes: {N_CLASSES}")

# Setup MPS with output dimension = N_CLASSES
L = 3
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

# Generate synthetic classification data
torch.manual_seed(42)
x_data = torch.randn(N_SAMPLES, DIM_PATCHES, DIM_PIXELS)
# Random class labels - convert to one-hot (N_SAMPLES, N_CLASSES)
y_labels = torch.randint(0, N_CLASSES, (N_SAMPLES,))
y_data = F.one_hot(y_labels, num_classes=N_CLASSES).float()

print(f"\nData:")
print(f"  Input shape: {x_data.shape}")
print(f"  Labels (one-hot) shape: {y_data.shape}")
print(f"  Label range: [0, {N_CLASSES-1}]")

input_labels_ntn = [
    [0, (f"{i}_patches", f"{i}_pixels")]
    for i in range(L)
]

loader = Inputs(
    inputs=[x_data],
    outputs=[y_data],
    outputs_labels=["out"],
    input_labels=input_labels_ntn,
    batch_dim="s",
    batch_size=BATCH_SIZE
)

loss = CrossEntropyLoss()
model = CMPO2_NTN(
    tn=tn,
    output_dims=["out"],
    input_dims=[str(i) for i in range(L)],
    loss=loss,
    data_stream=loader,
    cache_environments=True
)

print(f"\nModel:")
print(f"  Type: CMPO2_NTN")
print(f"  Loss: CrossEntropyLoss")
print(f"  Caching: {model.cache_environments}")

print(f"\n" + "-"*70)
print("TRAINING")
print("-"*70)

try:
    # Use classification metrics instead of regression
    metrics = model.fit(
        n_epochs=N_EPOCHS, 
        regularize=True, 
        jitter=1e-4, 
        verbose=True,
        eval_metrics=CLASSIFICATION_METRICS
    )
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    cache_stats = model.get_cache_stats()
    
    print(f"\n✓ Training completed successfully!")
    print(f"\nFinal Metrics:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    
    print(f"\nCache Performance:")
    print(f"  Cache hits: {cache_stats['hits']}")
    print(f"  Cache misses: {cache_stats['misses']}")
    print(f"  Total calls: {cache_stats['total']}")
    print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")
    
    # Random baseline accuracy
    random_acc = 1.0 / N_CLASSES
    print(f"\nBaseline:")
    print(f"  Random guessing: {random_acc:.2%}")
    
    if metrics['accuracy'] > random_acc * 1.5:
        print(f"\n✓✓✓ MODEL IS LEARNING! ✓✓✓")
        print(f"    Accuracy is {metrics['accuracy']/random_acc:.1f}x better than random!")
    else:
        print(f"\n⚠ Model barely better than random guessing")
        
except Exception as e:
    print(f"\n✗ Training failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
