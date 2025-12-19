# type: ignore
"""
Compare GTN (Gradient-based) vs NTN (Newton-based) on MNIST classification.
Running on CUDA if available.
"""
import torch
from model.GTN import GTN
from model.NTN import NTN
from model.builder import Inputs
from model.losses import CrossEntropyLoss
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
import quimb.tensor as qt
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 0. Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
torch.set_default_dtype(torch.float32)

print("="*70)
print(f"GTN vs NTN Comparison on MNIST | Device: {device}")
print("="*70)

# --- 1. Data Preprocessing ---
print("\nLoading and processing data...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load data (Download to CPU first)
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# Use smaller subset for faster testing
SUBSET_SIZE = 5000  
TEST_SIZE = 1000

train_loader_raw = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader_raw = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# Process Train
train_samples = []
train_labels = []
for images, labels in train_loader_raw:
    train_samples.append(images)
    train_labels.append(labels)
    if len(torch.cat(train_samples)) >= SUBSET_SIZE:
        break
        
xinp_train = torch.cat(train_samples, dim=0)[:SUBSET_SIZE]
y_train = torch.cat(train_labels, dim=0)[:SUBSET_SIZE]

# Process Test
test_samples = []
test_labels = []
for images, labels in test_loader_raw:
    test_samples.append(images)
    test_labels.append(labels)
    if len(torch.cat(test_samples)) >= TEST_SIZE:
        break
        
xinp_test = torch.cat(test_samples, dim=0)[:TEST_SIZE]
y_test = torch.cat(test_labels, dim=0)[:TEST_SIZE]

# Unfold images into patches
KERNEL_SIZE = 4
STRIDE = 4

xinp_train = F.unfold(xinp_train, kernel_size=(KERNEL_SIZE,KERNEL_SIZE), stride=(STRIDE,STRIDE), padding=0).transpose(-2, -1)
xinp_train = torch.cat((xinp_train, torch.zeros((xinp_train.shape[0], 1, xinp_train.shape[2]))), dim=-2)
xinp_train = torch.cat((xinp_train, torch.zeros((xinp_train.shape[0], xinp_train.shape[1], 1))), dim=-1)
xinp_train[..., -1, -1] = 1.0

xinp_test = F.unfold(xinp_test, kernel_size=(KERNEL_SIZE,KERNEL_SIZE), stride=(STRIDE,STRIDE), padding=0).transpose(-2, -1)
xinp_test = torch.cat((xinp_test, torch.zeros((xinp_test.shape[0], 1, xinp_test.shape[2]))), dim=-2)
xinp_test = torch.cat((xinp_test, torch.zeros((xinp_test.shape[0], xinp_test.shape[1], 1))), dim=-1)
xinp_test[..., -1, -1] = 1.0

# --- CRITICAL: Move Data to Device ---
xinp_train = xinp_train.to(device)
y_train = y_train.to(device)
xinp_test = xinp_test.to(device)
y_test = y_test.to(device)
# -------------------------------------

print(f"Train data shape: {xinp_train.shape}")
print(f"Test data shape: {xinp_test.shape}")
print(f"Number of classes: 10")

# --- 2. Define Model Architecture ---

class Conv(GTN):
    """GTN version with autograd"""
    def construct_nodes(self, x):
        input_nodes = []
        for i in self.input_dims:
            a = qt.Tensor(x, inds=["s", f"{i}_patches", f"{i}_pixels"], tags=f"Input_{i}")
            input_nodes.append(a)
        return input_nodes

# Derived Dimensions
DIM_PATCHES = xinp_train.shape[1] 
DIM_PIXELS = xinp_train.shape[2] 

bond_dim = 3

def init_data(*shape):
    """Initialize tensor with small random values ON DEVICE"""
    return torch.randn(*shape, device=device) * 0.1

def create_tn():
    """Create tensor network architecture (shared by both models)"""
    # Pixels MPS
    pixels_mps = [
        qt.Tensor(data=init_data(DIM_PIXELS, bond_dim), inds=["0_pixels", "bond_p_01"], tags=["0_Pi"]),
        qt.Tensor(data=init_data(bond_dim, DIM_PIXELS, bond_dim, 10), inds=["bond_p_01", "1_pixels", "bond_p_12", "class_out"], tags=["1_Pi"]),
        qt.Tensor(data=init_data(bond_dim, DIM_PIXELS), inds=["bond_p_12", "2_pixels"], tags=["2_Pi"])
    ]
    
    # Patches MPS
    patches_mps = [
        qt.Tensor(data=init_data(bond_dim, DIM_PATCHES, bond_dim), inds=["r1", "0_patches", "bond_pt_01"], tags=["0_Pa"]),
        qt.Tensor(data=init_data(bond_dim, DIM_PATCHES, bond_dim), inds=["bond_pt_01", "1_patches", "bond_pt_12"], tags=["1_Pa"]),
        qt.Tensor(data=init_data(bond_dim, DIM_PATCHES, bond_dim), inds=["bond_pt_12", "2_patches", "r1"], tags=["2_Pa"])
    ]
    
    return qt.TensorNetwork(patches_mps + pixels_mps)

input_labels = ["0", "1", "2"]

# --- 3. Setup GTN Model (Gradient-based) ---
print("\n" + "="*70)
print("Setting up GTN (Gradient-based with PyTorch autograd)")
print("="*70)

tn_gtn = create_tn()
model_gtn = Conv(
    tn=tn_gtn,
    output_dims=["s", "class_out"],
    input_dims=input_labels
)

model_gtn.to(device)

optimizer_gtn = optim.Adam(model_gtn.parameters(), lr=1e-2)
criterion_gtn = nn.CrossEntropyLoss()

BATCH_SIZE_GTN = 100
train_ds_gtn = torch.utils.data.TensorDataset(xinp_train, y_train)
test_ds_gtn = torch.utils.data.TensorDataset(xinp_test, y_test)
train_loader_gtn = torch.utils.data.DataLoader(train_ds_gtn, batch_size=BATCH_SIZE_GTN, shuffle=True)
test_loader_gtn = torch.utils.data.DataLoader(test_ds_gtn, batch_size=BATCH_SIZE_GTN, shuffle=False)

def evaluate_gtn(loader):
    model_gtn.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            output = model_gtn(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    model_gtn.train()
    return correct / total

# --- 4. Setup NTN Model (Newton-based) ---
print("\n" + "="*70)
print("Setting up NTN (Newton-based with 2nd order derivatives)")
print("="*70)

tn_ntn = create_tn() 

BATCH_SIZE_NTN = 500

input_labels_ntn = [
    [0, ("0_patches", "0_pixels")], 
    [0, ("1_patches", "1_pixels")],
    [0, ("2_patches", "2_pixels")]
]

loader_ntn = Inputs(
    inputs=[xinp_train],
    outputs=[y_train.unsqueeze(1)], 
    outputs_labels=["class_out"],
    input_labels=input_labels_ntn, 
    batch_dim="s",
    batch_size=BATCH_SIZE_NTN
)

loss_ntn = CrossEntropyLoss()
print(f"CrossEntropyLoss.use_diagonal_hessian = {loss_ntn.use_diagonal_hessian}")

model_ntn = NTN(
    tn=tn_ntn,
    output_dims=["class_out"],
    input_dims=input_labels, 
    loss=loss_ntn,
    data_stream=loader_ntn,
    method='cholesky'
)

# Create input tensors once on device
test_inputs_ntn = [qt.Tensor(xinp_test, inds=["s", f"{i}_patches", f"{i}_pixels"], tags=f"Input_{i}") 
                   for i in input_labels]

test_loader_ntn = Inputs(
    inputs=[xinp_test],
    outputs=[y_test.unsqueeze(1)], 
    outputs_labels=["class_out"],
    input_labels=input_labels_ntn, 
    batch_dim="s",
    batch_size=BATCH_SIZE_NTN 
)

def evaluate_ntn():
    """Evaluate NTN using the efficient generator forward pass"""
    # input_generator=test_loader_ntn.data_mu yields batches of input tensors.
    # model_ntn.forward handles the iteration and concatenation of results.
    
    logits_qt = model_ntn.forward(model_ntn.tn, input_generator=test_loader_ntn.data_mu)
    
    logits = logits_qt.data
    preds = torch.argmax(logits, dim=1)
    correct = (preds == y_test).sum().item()
    return correct / len(y_test)

# --- 5. Training Comparison ---
print("\n" + "="*70)
print("Starting Training Comparison")
print("="*70)

EPOCHS = 10

history = {
    'epoch': [],
    'gtn_loss': [],
    'gtn_acc': [],
    'ntn_acc': []
}

print(f"\nTraining for {EPOCHS} epochs...")
print(f"GTN: Batch size={BATCH_SIZE_GTN}, Optimizer=Adam(lr=1e-2)")
print(f"NTN: Batch size={BATCH_SIZE_NTN}, Method=Newton with Cholesky")

jitter = [2,1] + [1e-1]*100
for epoch in range(EPOCHS):
    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"{'='*70}")
    
    # === GTN Training ===
    model_gtn.train()
    gtn_running_loss = 0.0
    pbar = tqdm(train_loader_gtn, desc=f"GTN Epoch {epoch+1}")
    
    for batch_idx, (data, target) in enumerate(pbar):
        optimizer_gtn.zero_grad()
        output = model_gtn(data)
        loss = criterion_gtn(output, target)
        loss.backward()
        optimizer_gtn.step()
        
        gtn_running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    gtn_avg_loss = gtn_running_loss / len(train_loader_gtn)
    gtn_acc = evaluate_gtn(test_loader_gtn)
    
    # === NTN Training ===
    print(f"NTN Epoch {epoch+1}: Running Newton sweep...")
    from model.utils import CLASSIFICATION_METRICS
    ntn_metrics = model_ntn.fit(n_epochs=1, regularize=True, jitter=jitter[epoch], verbose=False, eval_metrics=CLASSIFICATION_METRICS)
    ntn_acc = evaluate_ntn()
    
    # Record history
    history['epoch'].append(epoch + 1)
    history['gtn_loss'].append(gtn_avg_loss)
    history['gtn_acc'].append(gtn_acc * 100)
    history['ntn_acc'].append(ntn_acc * 100)
    
    # Print comparison
    print(f"\nResults:")
    print(f"  GTN: Loss={gtn_avg_loss:.4f}, Test Acc={gtn_acc*100:.2f}%")
    print(f"  NTN: Test Acc={ntn_acc*100:.2f}%")

# --- 6. Final Comparison ---
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

print(f"\nFinal Test Accuracy:")
print(f"  GTN (Gradient-based):  {history['gtn_acc'][-1]:.2f}%")
print(f"  NTN (Newton-based):    {history['ntn_acc'][-1]:.2f}%")

# --- 7. Visualization ---
plt.figure(figsize=(12, 5))

# Subplot 1: GTN Loss
plt.subplot(1, 2, 1)
plt.plot(history['epoch'], history['gtn_loss'], 'r-o', label='GTN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GTN Training Loss')
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 2: Accuracy Comparison
plt.subplot(1, 2, 2)
plt.plot(history['epoch'], history['gtn_acc'], 'b-o', label='GTN Test Accuracy', linewidth=2)
plt.plot(history['epoch'], history['ntn_acc'], 'g-s', label='NTN Test Accuracy', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('GTN vs NTN Accuracy Comparison')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('gtn_vs_ntn_comparison.png', dpi=150)
print("\nPlot saved to 'gtn_vs_ntn_comparison.png'")

# Print summary table
print("\n" + "="*70)
print("EPOCH-BY-EPOCH COMPARISON")
print("="*70)
print(f"{'Epoch':<8} {'GTN Loss':<12} {'GTN Acc (%)':<12} {'NTN Acc (%)':<12} {'Difference':<12}")
print("-"*70)
for i in range(len(history['epoch'])):
    diff = history['ntn_acc'][i] - history['gtn_acc'][i]
    print(f"{history['epoch'][i]:<8} {history['gtn_loss'][i]:<12.4f} {history['gtn_acc'][i]:<12.2f} {history['ntn_acc'][i]:<12.2f} {diff:+.2f}%")

print("\n" + "="*70)
print("✓ Comparison complete!")
print("="*70)
