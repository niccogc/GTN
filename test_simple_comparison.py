# type: ignore
"""
Simple comparison: GTN vs NTN on regression task
Shows both methods work with same architecture and data.
"""
import torch
import quimb.tensor as qt
from model.GTN import GTN  
from model.NTN import NTN
from model.builder import Inputs
from model.losses import MSELoss
import torch.optim as optim
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

print("\n" + "="*70)
print("GTN vs NTN Comparison - Simple Regression")
print("="*70)

# === 1. Generate Data ===
print("\nGenerating data...")
N_SAMPLES = 1000
BATCH_SIZE = 200

x_raw = 2 * torch.rand(N_SAMPLES, 1) - 1
y_raw = x_raw**2 + 0.1 * torch.randn(N_SAMPLES, 1)  # y = x^2 + noise
x_features = torch.cat([x_raw, torch.ones_like(x_raw)], dim=1)  # [x, 1]

print(f"Data: X shape={x_features.shape}, Y shape={y_raw.shape}")

# === 2. Create Shared Architecture ===
D_bond = 4
D_phys = 2

def init_weights(shape):
    w = torch.randn(*shape) * 0.1
    return w / torch.norm(w)

def create_tn():
    """Shared MPS architecture"""
    t1 = qt.Tensor(data=init_weights((D_phys, D_bond)), inds=('x1', 'b1'), tags={'Node1'})
    t2 = qt.Tensor(data=init_weights((D_bond, D_phys, D_bond, 1)), inds=('b1', 'x2', 'b2','y'), tags={'Node2'})
    t3 = qt.Tensor(data=init_weights((D_bond, D_phys)), inds=('b2', 'x3'), tags={'Node3'})
    return qt.TensorNetwork([t1, t2, t3])

input_labels = ["x1", "x2", "x3"]

# === 3. Setup GTN (Gradient-based) ===
print("\n" + "="*70)
print("Setting up GTN (Gradient-based)")
print("="*70)

class SimpleGTN(GTN):
    def construct_nodes(self, x):
        input_nodes = []
        for i in self.input_dims:
            a = qt.Tensor(x, inds=["batch", f"{i}"], tags=f"Input_{i}")
            input_nodes.append(a)
        return input_nodes

tn_gtn = create_tn()
model_gtn = SimpleGTN(
    tn=tn_gtn,
    output_dims=["batch", "y"],
    input_dims=input_labels
)

optimizer_gtn = optim.Adam(model_gtn.parameters(), lr=0.01)
criterion_gtn = torch.nn.MSELoss()

# Create DataLoader
train_ds = torch.utils.data.TensorDataset(x_features, y_raw)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

def evaluate_gtn():
    model_gtn.eval()
    with torch.no_grad():
        y_pred = model_gtn(x_features)
        mse = torch.nn.functional.mse_loss(y_pred, y_raw).item()
    model_gtn.train()
    return mse

# === 4. Setup NTN (Newton-based) ===
print("\n" + "="*70)
print("Setting up NTN (Newton-based)")
print("="*70)

tn_ntn = create_tn()

loader_ntn = Inputs(
    inputs=[x_features],
    outputs=[y_raw],
    outputs_labels=["y"],
    input_labels=input_labels,
    batch_dim="batch",
    batch_size=BATCH_SIZE
)

loss_ntn = MSELoss()
print(f"MSELoss.use_diagonal_hessian = {loss_ntn.use_diagonal_hessian}")

model_ntn = NTN(
    tn=tn_ntn,
    output_dims=["y"],
    input_dims=input_labels,
    loss=loss_ntn,
    data_stream=loader_ntn,
    method='cholesky'
)

def evaluate_ntn():
    """Evaluate NTN MSE"""
    # Create input tensors
    inputs = [qt.Tensor(x_features, inds=["batch", f"{i}"], tags=f"Input_{i}") 
              for i in input_labels]
    
    # Forward pass
    output_tn = model_ntn.tn.copy()
    for inp in inputs:
        output_tn = output_tn & inp
    
    result = output_tn.contract(output_inds=["batch", "y"], optimize='auto-hq')
    y_pred = result.data
    mse = torch.nn.functional.mse_loss(y_pred, y_raw).item()
    return mse

# === 5. Training Comparison ===
print("\n" + "="*70)
print("Training Comparison")
print("="*70)

EPOCHS = 10

history = {
    'epoch': [],
    'gtn_mse': [],
    'ntn_mse': []
}

print(f"\nTraining for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-"*70)
    
    # === GTN Training ===
    model_gtn.train()
    gtn_loss_sum = 0
    for data, target in train_loader:
        optimizer_gtn.zero_grad()
        output = model_gtn(data)
        loss = criterion_gtn(output, target)
        loss.backward()
        optimizer_gtn.step()
        gtn_loss_sum += loss.item()
    
    gtn_mse = evaluate_gtn()
    
    # === NTN Training ===
    model_ntn.fit(n_epochs=1, regularize=True, jitter=1e-6, verbose=False)
    ntn_mse = evaluate_ntn()
    
    # Record
    history['epoch'].append(epoch + 1)
    history['gtn_mse'].append(gtn_mse)
    history['ntn_mse'].append(ntn_mse)
    
    print(f"  GTN: MSE = {gtn_mse:.6f}")
    print(f"  NTN: MSE = {ntn_mse:.6f}")

# === 6. Results ===
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

print(f"\nFinal MSE:")
print(f"  GTN (Gradient-based): {history['gtn_mse'][-1]:.6f}")
print(f"  NTN (Newton-based):   {history['ntn_mse'][-1]:.6f}")

# === 7. Visualization ===
plt.figure(figsize=(10, 5))
plt.plot(history['epoch'], history['gtn_mse'], 'b-o', label='GTN (Gradient)', linewidth=2)
plt.plot(history['epoch'], history['ntn_mse'], 'g-s', label='NTN (Newton)', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('GTN vs NTN: Training Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('gtn_vs_ntn_simple.png', dpi=150)
print("\nPlot saved to 'gtn_vs_ntn_simple.png'")

# Summary table
print("\n" + "="*70)
print("EPOCH-BY-EPOCH COMPARISON")
print("="*70)
print(f"{'Epoch':<8} {'GTN MSE':<15} {'NTN MSE':<15} {'Ratio (GTN/NTN)':<15}")
print("-"*70)
for i in range(len(history['epoch'])):
    ratio = history['gtn_mse'][i] / history['ntn_mse'][i]
    print(f"{history['epoch'][i]:<8} {history['gtn_mse'][i]:<15.6f} {history['ntn_mse'][i]:<15.6f} {ratio:<15.3f}")

print("\n" + "="*70)
print("✓ Comparison complete!")
print("="*70)
