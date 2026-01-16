#!/usr/bin/env python
"""
CUDA compatibility test for GTN/NTN models.
Run this on the cluster to verify GPU support works correctly.

Usage:
    python experiments/test_cuda.py
    python experiments/test_cuda.py --verbose
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

torch.set_default_dtype(torch.float64)


def print_header(msg):
    print(f"\n{'=' * 60}")
    print(f" {msg}")
    print(f"{'=' * 60}")


def print_result(name, passed, details=""):
    status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
    print(f"  [{status}] {name}")
    if details:
        print(f"         {details}")


def test_cuda_available():
    """Test 1: Basic CUDA availability."""
    print_header("Test 1: CUDA Availability")

    cuda_available = torch.cuda.is_available()
    print_result("torch.cuda.is_available()", cuda_available)

    if cuda_available:
        device_count = torch.cuda.device_count()
        print_result(f"GPU count: {device_count}", device_count > 0)

        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print_result(f"GPU {i}: {name}", True, f"Memory: {mem:.1f} GB")

        # Test basic tensor ops on CUDA
        x = torch.randn(100, 100, device="cuda")
        y = torch.randn(100, 100, device="cuda")
        z = x @ y
        print_result("Basic CUDA tensor matmul", z.device.type == "cuda")

    return cuda_available


def test_quimb_cuda():
    """Test 2: quimb TensorNetwork with CUDA tensors."""
    print_header("Test 2: quimb TensorNetwork on CUDA")

    try:
        import quimb.tensor as qt

        # Create a simple tensor network on CPU first
        A = qt.Tensor(torch.randn(4, 5), inds=["i", "j"], tags=["A"])
        B = qt.Tensor(torch.randn(5, 3), inds=["j", "k"], tags=["B"])
        tn = A & B

        # Move to CUDA using apply_to_arrays
        def to_cuda(x):
            if hasattr(x, "to"):
                return x.to("cuda")
            return x

        tn.apply_to_arrays(to_cuda)

        # Check tensors are on CUDA
        all_on_cuda = all(t.data.device.type == "cuda" for t in tn.tensors)
        print_result("TN tensors moved to CUDA", all_on_cuda)

        # Contract on CUDA
        result = tn.contract()
        print_result("TN contraction on CUDA", result.data.device.type == "cuda")
        print_result(f"Result shape: {result.data.shape}", True)

        return True

    except Exception as e:
        print_result("quimb CUDA test", False, str(e))
        return False


def test_gtn_cuda():
    """Test 3: GTN model forward pass on CUDA."""
    print_header("Test 3: GTN Model on CUDA")

    try:
        from model.GTN import GTN
        from model.MPO2_models import MPO2

        # Create a small MPO2 model
        L = 3
        bond_dim = 4
        phys_dim = 5
        output_dim = 2

        mpo = MPO2(
            L=L,
            bond_dim=bond_dim,
            phys_dim=phys_dim,
            output_dim=output_dim,
            output_site=1,
            init_strength=0.001,
        )

        # Wrap in GTN
        import quimb.tensor as qt

        class MPO2GTN(GTN):
            def construct_nodes(self, x):
                input_nodes = []
                for label in self.input_dims:
                    a = qt.Tensor(x, inds=["s", label], tags=f"Input_{label}")
                    input_nodes.append(a)
                return input_nodes

        gtn = MPO2GTN(tn=mpo.tn, output_dims=["out"], input_dims=mpo.input_dims)
        print_result("GTN model created", True)

        # Move to CUDA
        gtn = gtn.to("cuda")
        print_result("GTN moved to CUDA", True)

        # Check parameters are on CUDA
        params_on_cuda = all(p.device.type == "cuda" for p in gtn.parameters())
        print_result("GTN parameters on CUDA", params_on_cuda)

        # Forward pass with CUDA data
        batch_size = 8
        x = torch.randn(batch_size, phys_dim, device="cuda")
        output = gtn(x)

        output_on_cuda = output.device.type == "cuda"
        print_result("GTN forward pass on CUDA", output_on_cuda, f"Output shape: {output.shape}")

        # Backward pass
        loss = output.sum()
        loss.backward()
        print_result("GTN backward pass on CUDA", True)

        return True

    except Exception as e:
        import traceback

        print_result("GTN CUDA test", False, str(e))
        traceback.print_exc()
        return False


def test_typei_gtn_cuda():
    """Test 4: TypeI GTN models on CUDA."""
    print_header("Test 4: TypeI GTN Models on CUDA")

    try:
        from model.typeI import MPO2TypeI_GTN

        # Create TypeI model
        model = MPO2TypeI_GTN(
            max_sites=3,
            bond_dim=4,
            phys_dim=5,
            output_dim=2,
            output_site=1,
            init_strength=0.001,
        )
        print_result("MPO2TypeI_GTN created", True)

        # Move to CUDA
        model = model.to("cuda")
        print_result("Model moved to CUDA", True)

        # Forward pass
        batch_size = 8
        x = torch.randn(batch_size, 5, device="cuda")
        output = model(x)

        output_on_cuda = output.device.type == "cuda"
        print_result("Forward pass on CUDA", output_on_cuda, f"Output shape: {output.shape}")

        # Backward
        loss = output.sum()
        loss.backward()
        print_result("Backward pass on CUDA", True)

        return True

    except Exception as e:
        import traceback

        print_result("TypeI GTN CUDA test", False, str(e))
        traceback.print_exc()
        return False


def test_ntn_cuda():
    """Test 5: NTN (non-gradient) training on CUDA."""
    print_header("Test 5: NTN Model on CUDA")

    try:
        from model.standard import MPO2
        from model.base.NTN import NTN
        from model.losses import MSELoss
        from model.utils import create_inputs
        from experiments.device_utils import move_tn_to_device

        L = 3
        bond_dim = 4
        phys_dim = 5
        output_dim = 1

        mpo = MPO2(
            L=L,
            bond_dim=bond_dim,
            phys_dim=phys_dim,
            output_dim=output_dim,
            output_site=1,
            init_strength=0.1,
        )
        print_result("MPO2 model created", True)

        move_tn_to_device(mpo.tn)
        all_on_cuda = all(t.data.device.type == "cuda" for t in mpo.tn.tensors)
        print_result("TN moved to CUDA", all_on_cuda)

        n_samples = 64
        X = torch.randn(n_samples, phys_dim - 1, device="cuda")
        y = torch.randn(n_samples, 1, device="cuda")

        loader = create_inputs(
            X=X,
            y=y,
            input_labels=mpo.input_labels,
            output_labels=mpo.output_dims,
            batch_size=16,
            append_bias=True,
        )
        print_result("Data loader created on CUDA", True)

        ntn = NTN(
            tn=mpo.tn,
            output_dims=mpo.output_dims,
            input_dims=mpo.input_dims,
            loss=MSELoss(),
            data_stream=loader,
        )
        print_result("NTN created", True)

        from model.utils import REGRESSION_METRICS

        scores = ntn.evaluate(REGRESSION_METRICS, data_stream=loader)
        print_result("NTN evaluate on CUDA", True, f"Loss: {scores['loss']:.4f}")

        scores_train, scores_val = ntn.fit(
            n_epochs=2,
            regularize=True,
            jitter=1.0,
            verbose=False,
            eval_metrics=REGRESSION_METRICS,
        )
        print_result("NTN fit on CUDA", True, f"Final loss: {scores_train['loss']:.4f}")

        return True

    except Exception as e:
        import traceback

        print_result("NTN CUDA test", False, str(e))
        traceback.print_exc()
        return False


def test_training_loop():
    """Test 6: Full mini training loop on CUDA (GTN)."""
    print_header("Test 6: GTN Training Loop on CUDA")

    try:
        import torch.nn as nn
        import torch.optim as optim
        from model.typeI import MPO2TypeI_GTN

        device = torch.device("cuda")

        # Create model
        model = MPO2TypeI_GTN(
            max_sites=3,
            bond_dim=4,
            phys_dim=10,
            output_dim=3,
            output_site=1,
            init_strength=0.001,
        ).to(device)

        # Create dummy data
        n_samples = 64
        X = torch.randn(n_samples, 10, device=device)
        y = torch.randint(0, 3, (n_samples,), device=device)
        y_onehot = torch.zeros(n_samples, 3, device=device)
        y_onehot.scatter_(1, y.unsqueeze(1), 1)

        dataset = torch.utils.data.TensorDataset(X, y_onehot)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Train for a few steps
        model.train()
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print_result("Training loop completed", True, f"Avg loss: {total_loss / len(loader):.4f}")

        # Eval
        model.eval()
        with torch.no_grad():
            output = model(X)
            preds = output.argmax(dim=1)
            acc = (preds == y).float().mean().item()

        print_result("Evaluation completed", True, f"Accuracy: {acc:.2%}")

        return True

    except Exception as e:
        import traceback

        print_result("Training loop test", False, str(e))
        traceback.print_exc()
        return False


def test_memory_cleanup():
    """Test 7: CUDA memory management."""
    print_header("Test 7: CUDA Memory Management")

    try:
        # Get initial memory
        torch.cuda.empty_cache()
        initial_mem = torch.cuda.memory_allocated()

        # Allocate some tensors
        tensors = [torch.randn(1000, 1000, device="cuda") for _ in range(10)]
        peak_mem = torch.cuda.memory_allocated()

        # Delete and cleanup
        del tensors
        torch.cuda.empty_cache()
        final_mem = torch.cuda.memory_allocated()

        print_result(f"Initial memory: {initial_mem / 1e6:.1f} MB", True)
        print_result(f"Peak memory: {peak_mem / 1e6:.1f} MB", True)
        print_result(f"Final memory: {final_mem / 1e6:.1f} MB", final_mem <= initial_mem * 1.5)
        print_result("Memory cleanup works", final_mem < peak_mem)

        return True

    except Exception as e:
        print_result("Memory test", False, str(e))
        return False


def main():
    print("\n" + "=" * 60)
    print(" GTN/NTN CUDA Compatibility Test Suite")
    print("=" * 60)

    # Check CUDA first
    if not test_cuda_available():
        print("\n\033[91mCUDA not available. Cannot run GPU tests.\033[0m")
        print("Make sure you're on a GPU node with CUDA installed.")
        sys.exit(1)

    results = {
        "quimb_cuda": test_quimb_cuda(),
        "gtn_cuda": test_gtn_cuda(),
        "typei_gtn_cuda": test_typei_gtn_cuda(),
        "ntn_cuda": test_ntn_cuda(),
        "gtn_training_loop": test_training_loop(),
        "memory_cleanup": test_memory_cleanup(),
    }

    # Summary
    print_header("Summary")
    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "\033[92mPASS\033[0m" if result else "\033[91mFAIL\033[0m"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n\033[92mAll tests passed! CUDA support is working correctly.\033[0m")
        sys.exit(0)
    else:
        print("\n\033[91mSome tests failed. Check the output above.\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()
