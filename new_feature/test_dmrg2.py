import sys
sys.path.insert(0, '/home/nicci/Desktop/remote/GTN')

import torch
import quimb.tensor as qt
import numpy as np

from model.standard.TNML import TNML_P
from model.base.DMRG import DMRG
from model.utils import create_inputs
from model.losses import MSELoss


def create_test_tnml_model(n_features=5, bond_dim=7, output_dim=3):
    return TNML_P(
        L=5,
        bond_dim=bond_dim,
        phys_dim=n_features,
        output_dim=output_dim,
        output_site=n_features - 1,
    )


def create_test_data(n_samples=32, n_features=5, output_dim=3):
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, output_dim)
    return X, y


def test_get_trainable_node_pairs():
    print("=" * 60)
    print("TEST: _get_trainable_node_pairs")
    print("=" * 60)
    
    n_features = 5
    model = create_test_tnml_model(n_features=n_features)
    
    print(f"\nModel structure: {len(list(model.tn.tensors))} tensors")
    for tensor in model.tn:
        print(f"  {list(tensor.tags)[0]}: {tensor.inds} {tensor.shape}")
    
    X, y = create_test_data(n_features=n_features)
    loader = create_inputs(
        X=X, y=y,
        input_labels=model.input_labels,
        output_labels=model.output_dims,
        batch_size=16,
        append_bias=False,
        encoding="polynomial",
        poly_degree=model.poly_degree,
    )
    
    dmrg = DMRG(
        tn=model.tn,
        output_dims=model.output_dims,
        input_dims=model.input_dims,
        loss=MSELoss(),
        data_stream=loader,
    )
    
    pairs = dmrg._get_trainable_node_pairs()
    print(f"\nNode pairs:")
    for left, right in pairs:
        print(f"  ({left}, {right})")
    
    expected_pairs = [(f"Node{i}", f"Node{i+1}") for i in range(n_features - 1)]
    assert pairs == expected_pairs
    print(f"\n✓ PASS")
    
    return dmrg, model, loader


def test_get_bond_index_between(dmrg):
    print("\n" + "=" * 60)
    print("TEST: _get_bond_index_between")
    print("=" * 60)
    
    pairs = dmrg._get_trainable_node_pairs()
    
    for i, (left, right) in enumerate(pairs):
        bond_idx = dmrg._get_bond_index_between(left, right)
        print(f"  {left} <--[{bond_idx}]--> {right}")
        assert bond_idx == f"b{i}"
    
    print(f"\n✓ PASS")


def test_fuse_two_site_tensor(dmrg):
    print("\n" + "=" * 60)
    print("TEST: _fuse_two_site_tensor")
    print("=" * 60)
    
    pairs = dmrg._get_trainable_node_pairs()
    
    for tag_left, tag_right in pairs:
        fused, bond_idx, left_inds, right_inds = dmrg._fuse_two_site_tensor(tag_left, tag_right)
        
        expected_fused_inds = (set(left_inds) | set(right_inds)) - {bond_idx}
        
        print(f"  ({tag_left}, {tag_right}): {fused.inds} {fused.shape}")
        
        assert set(fused.inds) == expected_fused_inds
    
    print(f"\n✓ PASS")


def test_batch_environment(dmrg, loader):
    print("\n" + "=" * 60)
    print("TEST: _batch_environment (2-site)")
    print("=" * 60)
    
    pairs = dmrg._get_trainable_node_pairs()
    
    for inputs, y_true in loader.data_mu_y:
        print(f"\nBatch: y_true.shape={y_true.shape}")
        
        for tag_left, tag_right in pairs:
            tensor_left = dmrg.tn[tag_left]
            tensor_right = dmrg.tn[tag_right]
            bond_idx = dmrg._get_bond_index_between(tag_left, tag_right)
            
            env = dmrg._batch_environment(
                inputs, dmrg.tn, tag_left, tag_right,
                sum_over_batch=False, sum_over_output=False
            )
            
            left_external = set(tensor_left.inds) - {bond_idx}
            right_external = set(tensor_right.inds) - {bond_idx}
            fused_external = left_external | right_external
            
            expected_env_inds = {dmrg.batch_dim} | fused_external
            if 'out' in fused_external:
                expected_env_inds -= {'out'}
            else:
                expected_env_inds |= {'out'}
            
            print(f"  ({tag_left}, {tag_right}): env {env.inds} {env.shape}")
            
            assert set(env.inds) == expected_env_inds
        
        break
    
    print(f"\n✓ PASS")


def test_batch_node_derivatives(dmrg, loader):
    print("\n" + "=" * 60)
    print("TEST: _batch_node_derivatives (2-site)")
    print("=" * 60)
    
    pairs = dmrg._get_trainable_node_pairs()
    
    for inputs, y_true in loader.data_mu_y:
        print(f"\nBatch: y_true.shape={y_true.shape}")
        
        for tag_left, tag_right in pairs:
            grad, hess, fused_inds, bond_idx, left_inds, right_inds = \
                dmrg._batch_node_derivatives(inputs, y_true, tag_left, tag_right)
            
            fused_shape = tuple(
                dmrg.tn[tag_left].ind_size(i) if i in dmrg.tn[tag_left].inds 
                else dmrg.tn[tag_right].ind_size(i) 
                for i in fused_inds
            )
            
            print(f"  ({tag_left}, {tag_right}):")
            print(f"    fused_inds: {fused_inds}")
            print(f"    grad: {grad.inds} {grad.shape}")
            print(f"    hess: {hess.inds} {hess.shape}")
            
            assert set(grad.inds) == set(fused_inds), \
                f"Grad inds {grad.inds} != fused_inds {fused_inds}"
            assert grad.shape == fused_shape, \
                f"Grad shape {grad.shape} != expected {fused_shape}"
            
            hess_expected_inds = set(fused_inds) | {f"{x}_prime" for x in fused_inds}
            assert set(hess.inds) == hess_expected_inds, \
                f"Hess inds {set(hess.inds)} != expected {hess_expected_inds}"
        
        break
    
    print(f"\n✓ PASS")


def test_get_node_update(dmrg):
    print("\n" + "=" * 60)
    print("TEST: _get_node_update (2-site)")
    print("=" * 60)
    
    pairs = dmrg._get_trainable_node_pairs()
    
    for tag_left, tag_right in pairs:
        new_fused, bond_idx, left_inds, right_inds = dmrg._get_node_update(
            tag_left, tag_right, regularize=True, jitter=1e-4
        )
        
        original_fused, _, _, _ = dmrg._fuse_two_site_tensor(tag_left, tag_right)
        
        print(f"  ({tag_left}, {tag_right}):")
        print(f"    original fused: {original_fused.inds} {original_fused.shape}")
        print(f"    new fused: {new_fused.inds} {new_fused.shape}")
        print(f"    bond_idx: {bond_idx}")
        
        assert set(new_fused.inds) == set(original_fused.inds), \
            f"New fused inds {new_fused.inds} != original {original_fused.inds}"
        assert new_fused.shape == original_fused.shape, \
            f"New fused shape {new_fused.shape} != original {original_fused.shape}"
    
    print(f"\n✓ PASS")


def test_svd_split(dmrg):
    print("\n" + "=" * 60)
    print("TEST: _svd_split_two_site_tensor")
    print("=" * 60)
    
    pairs = dmrg._get_trainable_node_pairs()
    
    for tag_left, tag_right in pairs:
        fused, bond_idx, left_inds, right_inds = dmrg._fuse_two_site_tensor(
            tag_left, tag_right
        )
        
        original_left = dmrg.tn[tag_left]
        original_right = dmrg.tn[tag_right]
        
        new_left, new_right, new_bond_dim = dmrg._svd_split_two_site_tensor(
            fused, bond_idx, left_inds, right_inds,
            max_bond=None, cutoff=1e-10, absorb="right"
        )
        
        print(f"  ({tag_left}, {tag_right}):")
        print(f"    original left: {original_left.inds} {original_left.shape}")
        print(f"    new left: {new_left.inds} {new_left.shape}")
        print(f"    original right: {original_right.inds} {original_right.shape}")
        print(f"    new right: {new_right.inds} {new_right.shape}")
        print(f"    new bond dim: {new_bond_dim}")
        
        assert set(new_left.inds) == set(left_inds), \
            f"Left inds {new_left.inds} != expected {left_inds}"
        assert set(new_right.inds) == set(right_inds), \
            f"Right inds {new_right.inds} != expected {right_inds}"
        
        recontr = (new_left & new_right).contract()
        diff = torch.norm(recontr.data - fused.data) / torch.norm(fused.data)
        print(f"    reconstruction error: {diff:.2e}")
        assert diff < 1e-5, f"Reconstruction error {diff} too large"
    
    print(f"\n✓ PASS")


def test_svd_split_with_truncation(dmrg):
    print("\n" + "=" * 60)
    print("TEST: _svd_split_two_site_tensor (with truncation)")
    print("=" * 60)
    
    pairs = dmrg._get_trainable_node_pairs()
    
    for tag_left, tag_right in pairs:
        fused, bond_idx, left_inds, right_inds = dmrg._fuse_two_site_tensor(
            tag_left, tag_right
        )
        
        original_bond_dim = dmrg.tn[tag_left].ind_size(bond_idx) if bond_idx in dmrg.tn[tag_left].inds else dmrg.tn[tag_right].ind_size(bond_idx)
        
        new_left, new_right, new_bond_dim = dmrg._svd_split_two_site_tensor(
            fused, bond_idx, left_inds, right_inds,
            max_bond=3, cutoff=1e-10, absorb="right"
        )
        
        print(f"  ({tag_left}, {tag_right}):")
        print(f"    original bond dim: {original_bond_dim}")
        print(f"    truncated bond dim: {new_bond_dim}")
        print(f"    new left: {new_left.inds} {new_left.shape}")
        print(f"    new right: {new_right.inds} {new_right.shape}")
        
        assert new_bond_dim <= 3, f"Bond dim {new_bond_dim} > max_bond 3"
    
    print(f"\n✓ PASS")


def test_update_tn_node(dmrg, loader):
    print("\n" + "=" * 60)
    print("TEST: update_tn_node (full 2-site update)")
    print("=" * 60)
    
    pairs = dmrg._get_trainable_node_pairs()
    
    print("\nBefore update:")
    for tensor in dmrg.tn:
        print(f"  {list(tensor.tags)[0]}: {tensor.inds} {tensor.shape}")
    
    tag_left, tag_right = pairs[1]
    
    new_bond_dim = dmrg.update_tn_node(
        tag_left, tag_right,
        regularize=True, jitter=1e-4,
        max_bond=5, cutoff=1e-10, absorb="right"
    )
    
    print(f"\nAfter updating ({tag_left}, {tag_right}) with max_bond=5:")
    for tensor in dmrg.tn:
        print(f"  {list(tensor.tags)[0]}: {tensor.inds} {tensor.shape}")
    print(f"  New bond dim: {new_bond_dim}")
    
    bond_idx = dmrg._get_bond_index_between(tag_left, tag_right)
    assert dmrg.tn[tag_left].ind_size(bond_idx) == new_bond_dim
    assert dmrg.tn[tag_right].ind_size(bond_idx) == new_bond_dim
    assert new_bond_dim <= 5
    
    print(f"\n✓ PASS")


def test_fit():
    print("\n" + "=" * 60)
    print("TEST: fit (full 2-site DMRG training)")
    print("=" * 60)
    
    n_features = 5
    model = create_test_tnml_model(n_features=n_features, bond_dim=4, output_dim=3)
    
    X_train, y_train = create_test_data(n_samples=64, n_features=n_features, output_dim=3)
    X_val, y_val = create_test_data(n_samples=32, n_features=n_features, output_dim=3)
    
    loader_train = create_inputs(
        X=X_train, y=y_train,
        input_labels=model.input_labels,
        output_labels=model.output_dims,
        batch_size=16,
        append_bias=False,
        encoding="polynomial",
        poly_degree=model.poly_degree,
    )
    
    loader_val = create_inputs(
        X=X_val, y=y_val,
        input_labels=model.input_labels,
        output_labels=model.output_dims,
        batch_size=16,
        append_bias=False,
        encoding="polynomial",
        poly_degree=model.poly_degree,
    )
    
    dmrg = DMRG(
        tn=model.tn,
        output_dims=model.output_dims,
        input_dims=model.input_dims,
        loss=MSELoss(),
        data_stream=loader_train,
    )
    
    print("\nBefore training:")
    for tensor in dmrg.tn:
        print(f"  {list(tensor.tags)[0]}: {tensor.shape}")
    
    scores_train, scores_val = dmrg.fit(
        n_epochs=3,
        regularize=True,
        jitter=1e-4,
        verbose=True,
        val_data=loader_val,
        max_bond=8,
        cutoff=1e-10,
    )
    
    print("\nAfter training:")
    for tensor in dmrg.tn:
        print(f"  {list(tensor.tags)[0]}: {tensor.shape}")
    
    print(f"\nFinal train loss: {scores_train['loss']:.6f}")
    print(f"Final val loss: {scores_val['loss']:.6f}")
    
    print(f"\n✓ PASS")


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# DMRG2 Implementation Tests")
    print("#" * 60)
    
    dmrg, model, loader = test_get_trainable_node_pairs()
    test_get_bond_index_between(dmrg)
    test_fuse_two_site_tensor(dmrg)
    test_batch_environment(dmrg, loader)
    test_batch_node_derivatives(dmrg, loader)
    test_get_node_update(dmrg)
    test_svd_split(dmrg)
    test_svd_split_with_truncation(dmrg)
    test_update_tn_node(dmrg, loader)
    test_fit()
    
    print("\n" + "#" * 60)
    print("# All tests passed!")
    print("#" * 60)
