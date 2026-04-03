# type: ignore
"""
Test script to verify CPDA structure works with existing NTN/GTN infrastructure.

CPDA Structure:
- L nodes, each with shape (phys_dim, rank) and indices (x{i}, r)
- All nodes share the rank index "r"
- One node (output_site) has additional "out" index: (phys_dim, rank, out)
- Contraction should sum over all x{i} (via inputs) and r, leaving only (batch, out)
"""

import torch
import quimb.tensor as qt
import numpy as np


def create_cpda_tn(L: int, rank: int, phys_dim: int, output_dim: int, output_site: int = None):
    """
    Create a minimal CPDA tensor network.

    Args:
        L: Number of sites/features
        rank: CPD rank (shared across all nodes)
        phys_dim: Physical dimension per site
        output_dim: Output dimension
        output_site: Which node gets output (default: last)

    Returns:
        tn: TensorNetwork
        input_labels: List of input index names
        output_dims: List of output index names
    """
    if output_site is None:
        output_site = L - 1

    tensors = []
    for i in range(L):
        if i == output_site:
            # Output node: (phys_dim, rank, output_dim)
            shape = (phys_dim, rank, output_dim)
            inds = (f"x{i}", "r", "out")
        else:
            # Regular node: (phys_dim, rank)
            shape = (phys_dim, rank)
            inds = (f"x{i}", "r")

        data = torch.randn(*shape) * 0.1
        tensor = qt.Tensor(data=data, inds=inds, tags={f"Node{i}"})
        tensors.append(tensor)

    tn = qt.TensorNetwork(tensors)
    input_labels = [f"x{i}" for i in range(L)]
    output_dims = ["out"]

    return tn, input_labels, output_dims


def create_input_tensors(L: int, batch_size: int, phys_dim: int):
    """
    Create input tensors that match the CPDA structure.

    Each input tensor has shape (batch, phys_dim) with indices (s, x{i}).
    """
    inputs = []
    for i in range(L):
        data = torch.randn(batch_size, phys_dim)
        tensor = qt.Tensor(data=data, inds=("s", f"x{i}"), tags={f"Input{i}"})
        inputs.append(tensor)
    return inputs


def test_forward_pass():
    """Test that forward pass produces correct output shape."""
    print("=" * 60)
    print("TEST 1: Forward Pass")
    print("=" * 60)

    # Parameters
    L = 4
    rank = 3
    phys_dim = 5
    output_dim = 2
    batch_size = 10

    # Create CPDA TN
    tn, input_labels, output_dims = create_cpda_tn(L, rank, phys_dim, output_dim)

    print(f"CPDA Structure: L={L}, rank={rank}, phys_dim={phys_dim}, output_dim={output_dim}")
    print(f"Input labels: {input_labels}")
    print(f"Output dims: {output_dims}")
    print()

    # Print tensor info
    print("Tensors in TN:")
    for tensor in tn:
        print(f"  {list(tensor.tags)[0]}: shape={tensor.shape}, inds={tensor.inds}")
    print()

    # Create inputs
    inputs = create_input_tensors(L, batch_size, phys_dim)

    print("Input tensors:")
    for inp in inputs:
        print(f"  {list(inp.tags)[0]}: shape={inp.shape}, inds={inp.inds}")
    print()

    # Combine TN with inputs
    full_tn = tn & inputs

    print("Full TN outer indices:", list(full_tn.outer_inds()))
    print()

    # Contract
    output_inds = ["s"] + output_dims
    result = full_tn.contract(output_inds=output_inds)

    print(f"Result shape: {result.shape}")
    print(f"Result inds: {result.inds}")
    print(f"Expected shape: ({batch_size}, {output_dim})")

    assert result.shape == (batch_size, output_dim), (
        f"Shape mismatch! Got {result.shape}, expected ({batch_size}, {output_dim})"
    )
    assert result.inds == ("s", "out"), f"Index mismatch! Got {result.inds}, expected ('s', 'out')"

    print("PASSED!")
    print()
    return True


def test_environment_computation():
    """Test that environment computation produces correct shape for NTN."""
    print("=" * 60)
    print("TEST 2: Environment Computation")
    print("=" * 60)

    # Parameters
    L = 4
    rank = 3
    phys_dim = 5
    output_dim = 2
    batch_size = 10
    output_site = L - 1

    # Create CPDA TN
    tn, input_labels, output_dims = create_cpda_tn(L, rank, phys_dim, output_dim, output_site)

    # Create inputs
    inputs = create_input_tensors(L, batch_size, phys_dim)

    print(f"Testing environment for each node...")
    print()

    for target_idx in range(L):
        target_tag = f"Node{target_idx}"
        target_tensor = tn[target_tag]

        # Compute environment (mimicking NTN._batch_environment)
        env_tn = tn.copy() & inputs
        env_tn.delete(target_tag)

        outer_inds = list(env_tn.outer_inds())

        # Contract environment
        env_tensor = env_tn.contract(output_inds=outer_inds)

        print(f"Environment for {target_tag}:")
        print(f"  Target tensor shape: {target_tensor.shape}, inds: {target_tensor.inds}")
        print(f"  Env outer_inds: {outer_inds}")
        print(f"  Env tensor shape: {env_tensor.shape}, inds: {env_tensor.inds}")

        # Verify: env should have batch dim + target's indices (to contract with target)
        # For CPDA: env should have (s, x{i}, r) for non-output nodes
        #           and (s, x{i}, r, out) for output node
        expected_inds = ["s"] + list(target_tensor.inds)

        # Check all expected indices are present
        for exp_ind in expected_inds:
            assert exp_ind in env_tensor.inds, f"Missing index {exp_ind} in env for {target_tag}"

        # Verify forward from environment works
        forward_tn = env_tensor & target_tensor
        forward_result = forward_tn.contract(output_inds=["s", "out"])

        print(f"  Forward from env: shape={forward_result.shape}, inds={forward_result.inds}")
        assert forward_result.shape == (batch_size, output_dim), (
            f"Forward shape mismatch for {target_tag}"
        )
        print(f"  PASSED!")
        print()

    print("All environment tests PASSED!")
    print()
    return True


def test_environment_for_ntn_derivatives():
    """
    Test environment computation specifically for NTN derivative computation.

    In NTN, the environment is used to compute:
    - Forward: env @ node -> (batch, out)
    - Gradient: env * dL -> (node_inds)
    - Hessian: env * d2L * env' -> (node_inds, node_inds')
    """
    print("=" * 60)
    print("TEST 3: Environment for NTN Derivatives")
    print("=" * 60)

    L = 4
    rank = 3
    phys_dim = 5
    output_dim = 2
    batch_size = 10
    output_site = L - 1

    tn, input_labels, output_dims = create_cpda_tn(L, rank, phys_dim, output_dim, output_site)
    inputs = create_input_tensors(L, batch_size, phys_dim)

    # Test for output node (most complex case)
    target_tag = f"Node{output_site}"
    target_tensor = tn[target_tag]

    print(f"Testing NTN derivative computation for {target_tag}")
    print(f"Target tensor: shape={target_tensor.shape}, inds={target_tensor.inds}")
    print()

    # Compute environment (sum_over_batch=False, sum_over_output=False for derivatives)
    env_tn = tn.copy() & inputs
    env_tn.delete(target_tag)

    outer_inds = list(env_tn.outer_inds())
    env_tensor = env_tn.contract(output_inds=outer_inds)

    print(f"Environment: shape={env_tensor.shape}, inds={env_tensor.inds}")

    # Simulate dL_dy: (batch, out)
    dL_dy = qt.Tensor(data=torch.randn(batch_size, output_dim), inds=("s", "out"))

    # Compute gradient: sum over batch and output, leaving node indices
    node_inds = target_tensor.inds
    grad_tn = env_tensor & dL_dy
    node_grad = grad_tn.contract(output_inds=list(node_inds))

    print(f"Gradient: shape={node_grad.shape}, inds={node_grad.inds}")
    print(f"Expected shape: {target_tensor.shape}")

    assert node_grad.shape == target_tensor.shape, f"Gradient shape mismatch!"
    assert set(node_grad.inds) == set(target_tensor.inds), f"Gradient indices mismatch!"

    print("PASSED!")
    print()
    return True


def test_rank_index_behavior():
    """
    Test that the shared rank index 'r' behaves correctly.

    Key insight: In CPDA, all nodes share 'r'. When we remove one node,
    the remaining nodes still share 'r', which should remain as an outer index
    in the environment (not contracted away).
    """
    print("=" * 60)
    print("TEST 4: Rank Index Behavior")
    print("=" * 60)

    L = 3
    rank = 4
    phys_dim = 5
    output_dim = 2
    batch_size = 8

    tn, input_labels, output_dims = create_cpda_tn(L, rank, phys_dim, output_dim)
    inputs = create_input_tensors(L, batch_size, phys_dim)

    # Check that 'r' appears in all tensors
    print("Checking rank index 'r' in all tensors:")
    for tensor in tn:
        has_r = "r" in tensor.inds
        print(f"  {list(tensor.tags)[0]}: inds={tensor.inds}, has 'r': {has_r}")
        assert has_r, f"Missing 'r' index in {list(tensor.tags)[0]}"
    print()

    # When we remove Node0, the remaining nodes (Node1, Node2) still share 'r'
    # After contracting with inputs, 'r' should be the only shared internal index
    # that remains open in the environment

    target_tag = "Node0"
    env_tn = tn.copy() & inputs
    env_tn.delete(target_tag)

    print(f"After removing {target_tag}:")
    print(
        f"  Remaining tensors: {[list(t.tags)[0] for t in env_tn if 'Node' in str(list(t.tags)[0])]}"
    )
    print(f"  Outer indices: {list(env_tn.outer_inds())}")

    # The rank index should still be present (shared between remaining model nodes)
    # But it's an INTERNAL index between the remaining nodes, not outer
    # Unless there's only one remaining node that has 'r'

    env_tensor = env_tn.contract(output_inds=list(env_tn.outer_inds()))
    print(f"  Contracted env: shape={env_tensor.shape}, inds={env_tensor.inds}")

    # For Node0 (non-output): env should have (s, x0, r)
    # Because we need to contract with Node0's indices
    assert "r" in env_tensor.inds, "Rank index 'r' should be in environment!"
    assert "x0" in env_tensor.inds, "Physical index 'x0' should be in environment!"
    assert "s" in env_tensor.inds, "Batch index 's' should be in environment!"

    print("PASSED!")
    print()
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CPDA COMPATIBILITY TESTS")
    print("=" * 60 + "\n")

    all_passed = True

    try:
        all_passed &= test_forward_pass()
    except Exception as e:
        print(f"FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_environment_computation()
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_environment_for_ntn_derivatives()
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_rank_index_behavior()
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("CPDA structure is compatible with existing NTN/GTN infrastructure.")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)
