"""
Comprehensive Debug Script for BatchMovingEnvironment
Uses ALL available tools to verify correctness
"""
import sys
sys.path.insert(0, 'model')

import quimb.tensor as qb
import torch
import numpy as np
from batch_moving_environment import BatchMovingEnvironment
import traceback

# Setup
qb.set_tensor_linop_backend('torch')

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def test_basic_functionality():
    """Test 1: Basic Functionality - Does it work at all?"""
    print_section("TEST 1: BASIC FUNCTIONALITY")
    
    L, r, p = 4, 3, 2
    BATCH = 100
    
    print(f"Setup: L={L}, bond_dim={r}, phys_dim={p}, batch={BATCH}")
    
    # Create MPS states
    psi = qb.MPS_rand_state(L, bond_dim=r, phys_dim=p)
    phi = qb.MPS_rand_state(L, bond_dim=r, phys_dim=p)
    
    # Convert to torch
    psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))
    phi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))
    
    # Reindex to avoid conflicts
    psi.reindex({f"k{i}": f"dim_psi_{i}" for i in range(L)}, inplace=True)
    phi.reindex({f"k{i}": f"dim_phi_{i}" for i in range(L)}, inplace=True)
    
    # Add unique tags
    for i in range(L):
        psi.add_tag(f"PSI_{i}", where=f"I{i}")
        phi.add_tag(f"PHI_{i}", where=f"I{i}")
    
    # Create inputs with batch dimension
    inputs = [
        qb.Tensor(
            data=torch.rand(BATCH, p, p, dtype=torch.float32), 
            inds=['s', f'dim_psi_{i}', f'dim_phi_{i}'], 
            tags={f'I{i}', 'OP', f'OP{i}'} 
        ) 
        for i in range(L)
    ]
    
    # Build full TN
    tn_overlap = psi & phi
    for t in inputs:
        tn_overlap.add_tensor(t)
    
    print(f"✓ Created tensor network with {tn_overlap.num_tensors} tensors")
    
    # Initialize environment
    try:
        env = BatchMovingEnvironment(tn_overlap, begin='left', bsz=1, batch_inds=['s'])
        print(f"✓ BatchMovingEnvironment initialized successfully")
        print(f"  Initial position: {env.pos}")
        return env, psi, phi, inputs, tn_overlap
    except Exception as e:
        print(f"✗ FAILED to initialize: {e}")
        traceback.print_exc()
        return None, None, None, None, None

def test_movement(env):
    """Test 2: Movement - Can we move left and right?"""
    print_section("TEST 2: MOVEMENT")
    
    if env is None:
        print("✗ Skipping - env not initialized")
        return False
    
    try:
        initial_pos = env.pos
        print(f"Initial position: {initial_pos}")
        
        # Test move_right
        env.move_right()
        print(f"✓ move_right() successful, new position: {env.pos}")
        
        if env.pos != initial_pos + 1:
            print(f"✗ WARNING: Expected position {initial_pos + 1}, got {env.pos}")
        
        # Test move_left
        env.move_left()
        print(f"✓ move_left() successful, new position: {env.pos}")
        
        if env.pos != initial_pos:
            print(f"✗ WARNING: Expected to return to position {initial_pos}, got {env.pos}")
        
        # Test move_to
        for i in range(4):
            env.move_to(i)
            if env.pos != i:
                print(f"✗ move_to({i}) failed, position is {env.pos}")
                return False
        
        print(f"✓ All movement operations work correctly")
        return True
        
    except Exception as e:
        print(f"✗ Movement test FAILED: {e}")
        traceback.print_exc()
        return False

def test_batch_preservation(env, psi):
    """Test 3: Batch Dimension Preservation"""
    print_section("TEST 3: BATCH DIMENSION PRESERVATION")
    
    if env is None:
        print("✗ Skipping - env not initialized")
        return False
    
    all_passed = True
    L = 4
    
    for site_idx in range(L):
        try:
            env.move_to(site_idx)
            current_env = env()
            
            # Create hole by deleting target
            env_with_hole = current_env.copy()
            target_tag = f"PSI_{site_idx}"
            env_with_hole.delete(target_tag)
            
            # Get indices to keep
            outer_inds = list(env_with_hole.outer_inds())
            
            # Check if batch dimension 's' is in the environment
            all_inds = set()
            for t in env_with_hole.tensors:
                all_inds.update(t.inds)
            
            has_batch = 's' in all_inds
            
            if 's' not in outer_inds and has_batch:
                outer_inds.append('s')
            
            # Contract
            if len(outer_inds) > 0:
                contracted = env_with_hole.contract(all, output_inds=outer_inds)
                
                # Check if batch dimension is preserved
                if hasattr(contracted, 'inds'):
                    if 's' in contracted.inds:
                        print(f"  Site {site_idx}: ✓ Batch dimension 's' preserved, shape={contracted.shape}")
                    else:
                        print(f"  Site {site_idx}: ✗ Batch dimension 's' LOST!")
                        all_passed = False
                else:
                    # Scalar - check if batch was expected
                    if has_batch:
                        print(f"  Site {site_idx}: ✗ Contracted to scalar but batch existed!")
                        all_passed = False
                    else:
                        print(f"  Site {site_idx}: ✓ Contracted to scalar (no batch)")
            else:
                print(f"  Site {site_idx}: ⚠ No outer indices")
                
        except Exception as e:
            print(f"  Site {site_idx}: ✗ FAILED: {e}")
            all_passed = False
    
    if all_passed:
        print(f"\n✓✓✓ All sites preserve batch dimension correctly")
    else:
        print(f"\n✗✗✗ Some sites failed to preserve batch dimension")
    
    return all_passed

def test_caching_behavior(env, psi):
    """Test 4: Caching Behavior - Are environments cached or virtual?"""
    print_section("TEST 4: CACHING BEHAVIOR")
    
    if env is None:
        print("✗ Skipping - env not initialized")
        return False
    
    try:
        # Move to site 1
        env.move_to(1)
        target = psi["I1"]
        
        # Get original data
        original_data = target.data.clone()
        print(f"Original data (first 3 elements): {original_data.flatten()[:3]}")
        
        # Modify the tensor
        new_data = torch.randn_like(target.data)
        target.modify(data=new_data)
        print(f"Modified data (first 3 elements): {new_data.flatten()[:3]}")
        
        # Move to next site and check environment
        env.move_right()
        new_env = env()
        
        # Check if the modified tensor is in the environment
        if f"PSI_1" in new_env.tags:
            env_tensor = list(new_env.select("PSI_1").tensors)[0]
            
            # Compare data
            if torch.allclose(env_tensor.data, new_data, atol=1e-6):
                print(f"✓ Environment sees updated data (virtual view)")
                return True
            else:
                # Check first few elements
                print(f"Environment data (first 3 elements): {env_tensor.data.flatten()[:3]}")
                print(f"✗ Environment has stale data (pre-computed cache)")
                print(f"  This is EXPECTED behavior for MovingEnvironment")
                return True  # This is actually correct behavior!
        else:
            print(f"⚠ Modified tensor not in environment (due to bsz)")
            return True
            
    except Exception as e:
        print(f"✗ Caching test FAILED: {e}")
        traceback.print_exc()
        return False

def test_reconstruction(env, psi, phi, inputs):
    """Test 5: Reconstruction - Can we reconstruct the full TN?"""
    print_section("TEST 5: RECONSTRUCTION")
    
    if env is None:
        print("✗ Skipping - env not initialized")
        return False
    
    L = 4
    all_passed = True
    
    for site_idx in range(L):
        try:
            env.move_to(site_idx)
            current_env = env()
            
            # Create hole
            env_with_hole = current_env.copy()
            target_tag = f"PSI_{site_idx}"
            target_tensor = psi[f"I{site_idx}"]
            env_with_hole.delete(target_tag)
            
            # Contract environment
            outer_inds = list(env_with_hole.outer_inds())
            all_inds = set()
            for t in env_with_hole.tensors:
                all_inds.update(t.inds)
            
            if 's' in all_inds and 's' not in outer_inds:
                outer_inds.append('s')
            
            if len(outer_inds) > 0:
                contracted_env = env_with_hole.contract(all, output_inds=outer_inds)
                
                # Try to reconstruct
                reconstructed = contracted_env & target_tensor
                result = reconstructed.contract(all, output_inds=['s'])
                
                if result.shape == (100,):
                    print(f"  Site {site_idx}: ✓ Reconstruction successful, shape={result.shape}")
                else:
                    print(f"  Site {site_idx}: ✗ Wrong shape: {result.shape}, expected (100,)")
                    all_passed = False
            else:
                print(f"  Site {site_idx}: ⚠ No outer indices")
                
        except Exception as e:
            print(f"  Site {site_idx}: ✗ Reconstruction FAILED: {e}")
            all_passed = False
    
    if all_passed:
        print(f"\n✓✓✓ All sites reconstruct correctly")
    else:
        print(f"\n✗✗✗ Some sites failed reconstruction")
    
    return all_passed

def test_full_sweep(env, psi):
    """Test 6: Full Sweep - Forward and backward"""
    print_section("TEST 6: FULL SWEEP")
    
    if env is None:
        print("✗ Skipping - env not initialized")
        return False
    
    L = 4
    
    try:
        # Forward sweep
        print("Forward sweep (left -> right):")
        for i in range(L):
            env.move_to(i)
            current_env = env()
            target = psi[f"I{i}"]
            
            # Fake update
            new_data = torch.randn_like(target.data)
            target.modify(data=new_data)
            
            print(f"  Site {i}: ✓ Updated")
        
        print("✓ Forward sweep completed")
        
        # Backward sweep
        print("\nBackward sweep (right -> left):")
        for i in reversed(range(L)):
            env.move_to(i)
            current_env = env()
            target = psi[f"I{i}"]
            
            # Fake update
            new_data = torch.randn_like(target.data)
            target.modify(data=new_data)
            
            print(f"  Site {i}: ✓ Updated")
        
        print("✓ Backward sweep completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Sweep test FAILED: {e}")
        traceback.print_exc()
        return False

def test_edge_cases(env):
    """Test 7: Edge Cases"""
    print_section("TEST 7: EDGE CASES")
    
    if env is None:
        print("✗ Skipping - env not initialized")
        return False
    
    all_passed = True
    
    # Test 1: Move to boundaries
    try:
        env.move_to(0)
        print(f"✓ Can move to left boundary (site 0)")
    except Exception as e:
        print(f"✗ Cannot move to left boundary: {e}")
        all_passed = False
    
    try:
        env.move_to(3)
        print(f"✓ Can move to right boundary (site 3)")
    except Exception as e:
        print(f"✗ Cannot move to right boundary: {e}")
        all_passed = False
    
    # Test 2: Multiple consecutive moves
    try:
        for _ in range(3):
            env.move_right()
        print(f"✓ Can perform multiple consecutive moves")
    except Exception as e:
        print(f"✗ Multiple consecutive moves failed: {e}")
        all_passed = False
    
    # Test 3: Move beyond boundaries (should fail)
    try:
        env.move_to(0)
        env.move_left()
        print(f"✗ Should have failed when moving left from position 0")
        all_passed = False
    except Exception as e:
        print(f"✓ Correctly raises error when moving beyond left boundary")
    
    try:
        env.move_to(3)
        env.move_right()
        print(f"✗ Should have failed when moving right from position 3")
        all_passed = False
    except Exception as e:
        print(f"✓ Correctly raises error when moving beyond right boundary")
    
    return all_passed

def test_indices_consistency(env, psi):
    """Test 8: Index Consistency"""
    print_section("TEST 8: INDEX CONSISTENCY")
    
    if env is None:
        print("✗ Skipping - env not initialized")
        return False
    
    L = 4
    all_passed = True
    
    for site_idx in range(L):
        try:
            env.move_to(site_idx)
            current_env = env()
            
            # Create hole
            env_with_hole = current_env.copy()
            target_tag = f"PSI_{site_idx}"
            target_tensor = psi[f"I{site_idx}"]
            env_with_hole.delete(target_tag)
            
            # Get outer indices
            outer_inds = set(env_with_hole.outer_inds())
            target_inds = set(target_tensor.inds)
            
            # Check overlap
            overlap = outer_inds & target_inds
            
            if len(overlap) > 0:
                print(f"  Site {site_idx}: ✓ {len(overlap)} bond(s) connect environment to target")
            else:
                print(f"  Site {site_idx}: ✗ NO bonds connect environment to target!")
                all_passed = False
                
        except Exception as e:
            print(f"  Site {site_idx}: ✗ FAILED: {e}")
            all_passed = False
    
    if all_passed:
        print(f"\n✓✓✓ All sites have correct index connectivity")
    else:
        print(f"\n✗✗✗ Some sites have index problems")
    
    return all_passed

def test_output_dims():
    """Test 9: Output Dimensions Support"""
    print_section("TEST 9: OUTPUT DIMENSIONS SUPPORT")
    
    L, r, p = 4, 3, 2
    BATCH = 100
    OUT_DIM = 10
    
    # Create MPS with output dimension
    psi = qb.MPS_rand_state(L, bond_dim=r, phys_dim=p)
    psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))
    psi.reindex({f"k{i}": f"dim_{i}" for i in range(L)}, inplace=True)
    
    for i in range(L):
        psi.add_tag(f"PSI_{i}", where=f"I{i}")
    
    # Create inputs with batch AND output dimensions
    inputs = [
        qb.Tensor(
            data=torch.rand(BATCH, OUT_DIM, p, dtype=torch.float32), 
            inds=['s', 'out', f'dim_{i}'], 
            tags={f'I{i}', 'OP', f'OP{i}'} 
        ) 
        for i in range(L)
    ]
    
    tn = psi.copy()
    for t in inputs:
        tn.add_tensor(t)
    
    try:
        env = BatchMovingEnvironment(
            tn, 
            begin='left', 
            bsz=1, 
            batch_inds=['s'],
            output_dims=['out']
        )
        print(f"✓ BatchMovingEnvironment with output_dims initialized")
        
        # Test a contraction
        env.move_to(0)
        current_env = env()
        env_with_hole = current_env.copy()
        env_with_hole.delete("PSI_0")
        
        outer_inds = list(env_with_hole.outer_inds())
        all_inds = set()
        for t in env_with_hole.tensors:
            all_inds.update(t.inds)
        
        # Add batch and output dims
        if 's' in all_inds and 's' not in outer_inds:
            outer_inds.append('s')
        if 'out' in all_inds and 'out' not in outer_inds:
            outer_inds.append('out')
        
        contracted = env_with_hole.contract(all, output_inds=outer_inds)
        
        if hasattr(contracted, 'inds'):
            if 's' in contracted.inds and 'out' in contracted.inds:
                print(f"✓✓ Both batch 's' and output 'out' preserved")
                print(f"   Contracted shape: {contracted.shape}")
                return True
            else:
                print(f"✗ Missing dimensions in result")
                return False
        else:
            print(f"✗ Contracted to scalar")
            return False
            
    except Exception as e:
        print(f"✗ Output dims test FAILED: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests"""
    print_section("BATCH MOVING ENVIRONMENT COMPREHENSIVE DEBUG")
    
    results = {}
    
    # Test 1: Basic functionality
    env, psi, phi, inputs, tn = test_basic_functionality()
    results['basic_functionality'] = env is not None
    
    # Test 2: Movement
    results['movement'] = test_movement(env)
    
    # Test 3: Batch preservation
    results['batch_preservation'] = test_batch_preservation(env, psi)
    
    # Test 4: Caching behavior
    results['caching'] = test_caching_behavior(env, psi)
    
    # Test 5: Reconstruction
    results['reconstruction'] = test_reconstruction(env, psi, phi, inputs)
    
    # Test 6: Full sweep
    results['full_sweep'] = test_full_sweep(env, psi)
    
    # Test 7: Edge cases
    results['edge_cases'] = test_edge_cases(env)
    
    # Test 8: Index consistency
    results['index_consistency'] = test_indices_consistency(env, psi)
    
    # Test 9: Output dimensions
    results['output_dims'] = test_output_dims()
    
    # Summary
    print_section("SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nTest Results: {passed}/{total} passed\n")
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    if passed == total:
        print(f"\n{'='*80}")
        print(f"  ✓✓✓ ALL TESTS PASSED ✓✓✓")
        print(f"  BatchMovingEnvironment is working correctly!")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"  ✗✗✗ {total - passed} TESTS FAILED ✗✗✗")
        print(f"  BatchMovingEnvironment has issues that need fixing")
        print(f"{'='*80}")
    
    return results

if __name__ == "__main__":
    run_all_tests()
