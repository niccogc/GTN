# type: ignore
"""Test script for PartitionRank3 model."""

import torch
import numpy as np


def test_partition_rank3_structure():
    """Test that PartitionRank3 creates correct tensor network structure."""
    from model.partition_rank import PartitionRank3

    model = PartitionRank3(
        phys_dim=10,
        rank_dim=5,
        output_dim=3,
        use_tn_normalization=False,
    )

    assert len(model.tns) == 3, f"Expected 3 TNs, got {len(model.tns)}"
    assert len(model.input_dims_list) == 3
    assert len(model.input_labels_list) == 3
    assert model.output_dims == ["out"]

    for i, tn in enumerate(model.tns):
        assert len(list(tn.tensors)) == 2, f"TN {i} should have 2 tensors"

    tn_a = model.tns[0]
    a1 = tn_a["A1"]
    a2 = tn_a["A2"]
    assert set(a1.inds) == {"x0", "ra", "out"}, f"A1 has wrong indices: {a1.inds}"
    assert set(a2.inds) == {"ra", "x1", "x2"}, f"A2 has wrong indices: {a2.inds}"

    tn_b = model.tns[1]
    b1 = tn_b["B1"]
    b2 = tn_b["B2"]
    assert set(b1.inds) == {"x1", "rb", "out"}, f"B1 has wrong indices: {b1.inds}"
    assert set(b2.inds) == {"rb", "x0", "x2"}, f"B2 has wrong indices: {b2.inds}"

    tn_c = model.tns[2]
    c1 = tn_c["C1"]
    c2 = tn_c["C2"]
    assert set(c1.inds) == {"x2", "rc", "out"}, f"C1 has wrong indices: {c1.inds}"
    assert set(c2.inds) == {"rc", "x0", "x1"}, f"C2 has wrong indices: {c2.inds}"

    print("Structure test passed!")


def test_partition_rank3_contraction():
    """Test that each TN contracts to the expected output shape."""
    from model.partition_rank import PartitionRank3
    import quimb.tensor as qt

    phys_dim = 10
    rank_dim = 5
    output_dim = 3
    batch_size = 8

    model = PartitionRank3(
        phys_dim=phys_dim,
        rank_dim=rank_dim,
        output_dim=output_dim,
        use_tn_normalization=False,
    )

    x0_data = torch.randn(batch_size, phys_dim)
    x1_data = torch.randn(batch_size, phys_dim)
    x2_data = torch.randn(batch_size, phys_dim)

    inputs = [
        qt.Tensor(data=x0_data, inds=("s", "x0"), tags={"input_x0"}),
        qt.Tensor(data=x1_data, inds=("s", "x1"), tags={"input_x1"}),
        qt.Tensor(data=x2_data, inds=("s", "x2"), tags={"input_x2"}),
    ]

    total_output = None
    for i, tn in enumerate(model.tns):
        full_tn = tn & inputs
        result = full_tn.contract(output_inds=["s", "out"])
        assert result.shape == (batch_size, output_dim), (
            f"TN {i} wrong output shape: {result.shape}"
        )

        if total_output is None:
            total_output = result.data
        else:
            total_output = total_output + result.data

    assert total_output.shape == (batch_size, output_dim)
    print(f"Contraction test passed! Output shape: {total_output.shape}")


def test_partition_rank3_with_ntn_ensemble():
    """Test PartitionRank3 with NTN_Ensemble for training."""
    from model.partition_rank import PartitionRank3
    from model.base.NTN_Ensemble import NTN_Ensemble
    from model.losses import MSELoss

    phys_dim = 10
    rank_dim = 5
    output_dim = 1
    n_samples = 100
    batch_size = 32

    model = PartitionRank3(
        phys_dim=phys_dim,
        rank_dim=rank_dim,
        output_dim=output_dim,
        use_tn_normalization=False,
    )

    X_train = torch.randn(n_samples, phys_dim)
    y_train = torch.randn(n_samples, output_dim)

    loss = MSELoss()

    ntn = NTN_Ensemble(
        tns=model.tns,
        input_dims_list=model.input_dims_list,
        input_labels_list=model.input_labels_list,
        output_dims=model.output_dims,
        loss=loss,
        X_train=X_train,
        y_train=y_train,
        batch_size=batch_size,
    )

    assert ntn.n_models == 3
    print(f"NTN_Ensemble created with {ntn.n_models} models")

    scores_train, scores_val = ntn.fit(n_epochs=2, verbose=True)

    assert "loss" in scores_train
    assert "quality" in scores_train
    print(
        f"Training completed! Final train loss: {scores_train['loss']:.6f}, quality: {scores_train['quality']}"
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Testing PartitionRank3 Model")
    print("=" * 60)

    print("\n1. Testing structure...")
    test_partition_rank3_structure()

    print("\n2. Testing contraction...")
    test_partition_rank3_contraction()

    print("\n3. Testing with NTN_Ensemble...")
    test_partition_rank3_with_ntn_ensemble()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
