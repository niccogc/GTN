# type: ignore
"""
Test what tn.copy() actually does
"""
import torch
import quimb.tensor as qt

print("="*70)
print("Understanding TensorNetwork.copy()")
print("="*70)

# Create a simple TN
tn = qt.TensorNetwork([])
t1 = qt.Tensor(torch.randn(2, 3), inds=('a', 'b'), tags=['T1'])
tn.add_tensor(t1)

print("\nOriginal TN:")
print(f"  TN object ID: {id(tn)}")
print(f"  Tensor T1 object ID: {id(tn['T1'])}")
print(f"  Tensor T1 data ID: {id(tn['T1'].data)}")
print(f"  Tensor T1 norm: {torch.norm(tn['T1'].data):.6f}")

# Copy the TN
tn_copy = tn.copy()

print("\nCopied TN:")
print(f"  TN object ID: {id(tn_copy)}")
print(f"  Tensor T1 object ID: {id(tn_copy['T1'])}")
print(f"  Tensor T1 data ID: {id(tn_copy['T1'].data)}")
print(f"  Tensor T1 norm: {torch.norm(tn_copy['T1'].data):.6f}")

print("\nComparison:")
print(f"  Same TN object? {id(tn) == id(tn_copy)}")
print(f"  Same Tensor object? {id(tn['T1']) == id(tn_copy['T1'])}")
print(f"  Same data array? {id(tn['T1'].data) == id(tn_copy['T1'].data)}")

# Now update the original tensor
print("\n" + "="*70)
print("Updating original TN")
print("="*70)

# Delete and replace
tn.delete('T1')
t1_new = qt.Tensor(torch.ones(2, 3), inds=('a', 'b'), tags=['T1'])
tn.add_tensor(t1_new)

print("\nAfter update in original TN:")
print(f"  Original TN Tensor T1 object ID: {id(tn['T1'])}")
print(f"  Original TN Tensor T1 data ID: {id(tn['T1'].data)}")
print(f"  Original TN Tensor T1 norm: {torch.norm(tn['T1'].data):.6f}")

print(f"\n  Copied TN Tensor T1 object ID: {id(tn_copy['T1'])}")
print(f"  Copied TN Tensor T1 data ID: {id(tn_copy['T1'].data)}")
print(f"  Copied TN Tensor T1 norm: {torch.norm(tn_copy['T1'].data):.6f}")

print("\nResult:")
if torch.norm(tn['T1'].data - tn_copy['T1'].data) > 0.1:
    print("  ✓ Copied TN still has OLD tensor (shallow copy)")
    print("  ✓ This means cached environments WILL be stale")
else:
    print("  ✗ Copied TN somehow got updated too")

print("\n" + "="*70)
