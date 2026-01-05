# type: ignore
"""
Test with more stable training settings
"""
import torch
import torch.nn.functional as F
import quimb.tensor as qt
from model.MPS import CMPO2_NTN
from model.builder import Inputs
from model.losses import CrossEntropyLoss
from model.utils import CLASSIFICATION_METRICS

print("Testing CMPO2 with stable settings...")

BATCH_SIZE = 20
N_SAMPLES = 100  
BOND_DIM = 2  # Smaller
N_CLASSES = 3  # Fewer classes
N_EPOCHS = 3

L = 3
psi = qt.MPS_rand_state(L, bond_dim=BOND_DIM, phys_dim=4)
phi = qt.MPS_rand_state(L, bond_dim=BOND_DIM, phys_dim=5)

middle_psi = psi['I1']
middle_psi.new_ind('out', size=N_CLASSES, axis=-1, mode='random', rand_strength=0.001)  # Very small init
psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))
phi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))

psi.reindex({f"k{i}": f"{i}_pixels" for i in range(L)}, inplace=True)
phi.reindex({f"k{i}": f"{i}_patches" for i in range(L)}, inplace=True)

for i in range(L):
    psi.add_tag(f"{i}_Pi", where=f"I{i}")
    phi.add_tag(f"{i}_Pa", where=f"I{i}")

tn = psi & phi

torch.manual_seed(42)
x_data = torch.randn(N_SAMPLES, 5, 4) * 0.1  # Smaller inputs
y_labels = torch.randint(0, N_CLASSES, (N_SAMPLES,))
y_data = F.one_hot(y_labels, num_classes=N_CLASSES).float()

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

print(f"Settings: {N_CLASSES} classes, bond_dim={BOND_DIM}, jitter=0.1 (high regularization)")

try:
    metrics = model.fit(
        n_epochs=N_EPOCHS, 
        regularize=True, 
        jitter=0.1,  # MUCH higher regularization
        verbose=True,
        eval_metrics=CLASSIFICATION_METRICS
    )
    
    print(f"\nFinal: Loss={metrics['loss']:.2f}, Accuracy={metrics['accuracy']:.2%}")
    print(f"Random baseline: {1.0/N_CLASSES:.2%}")
    
    if metrics['accuracy'] > 1.5 / N_CLASSES:
        print("✓ Learning!")
    else:
        print("✗ Not learning")
        
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
