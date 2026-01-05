# Analyze correct caching strategy

print("="*70)
print("CACHING STRATEGY ANALYSIS")
print("="*70)

print("\nCurrent implementation:")
print("  - Cache key = id(inputs[0]) (per batch)")
print("  - Cache created on first call with a batch")
print("  - Cache reused for same batch across different nodes")
print("")

print("Scenario: Update nodes [0_Pi, 0_Pa, 1_Pi, 1_Pa, 2_Pi, 2_Pa]")
print("         Process batches [B1, B2, B3, B4, B5]")
print("")

print("What happens:")
print("  Node 0_Pi:")
print("    B1: MISS (create cache for B1 with current TN state)")
print("    B2: MISS (create cache for B2 with current TN state)")
print("    B3: MISS (create cache for B3 with current TN state)")
print("    B4: MISS (create cache for B4 with current TN state)")
print("    B5: MISS (create cache for B5 with current TN state)")
print("    --> 5 misses")
print("")

print("  UPDATE 0_Pi --> TN changes")
print("")

print("  Node 0_Pa:")
print("    B1: HIT (reuse cache for B1, but TN has changed!)")
print("    B2: HIT (reuse cache for B2, but TN has changed!)")
print("    B3: HIT (reuse cache for B3, but TN has changed!)")
print("    B4: HIT (reuse cache for B4, but TN has changed!)")
print("    B5: HIT (reuse cache for B5, but TN has changed!)")
print("    --> 5 hits, but STALE!")
print("")

print("  UPDATE 0_Pa --> TN changes again")
print("")

print("  Node 1_Pi:")
print("    B1: HIT (reuse cache for B1, but 0_Pi and 0_Pa changed!)")
print("    ... (all stale)")
print("")

print("="*70)
print("PROBLEM: Cached environments have OLD tensor values")
print("="*70)
print("")

print("The cached BatchMovingEnvironment contains:")
print("  full_tn = tn.copy()  <-- This was a COPY of tn")
print("  The copy contains tensor OBJECTS from the original tn")
print("")

print("When we update a node:")
print("  1. model.tn.delete(node_tag)")
print("  2. model.tn.add_tensor(new_tensor)")
print("")

print("But the cached environment still has:")
print("  - The OLD tensor object for that node")
print("  - Because it's in a COPIED TN")
print("")

print("="*70)
print("SOLUTIONS:")
print("="*70)
print("")

print("Option 1: Clear cache after EACH node update")
print("  Pro: Always fresh environments")
print("  Con: No caching benefit (0% hit rate)")
print("")

print("Option 2: Clear cache after EACH column update")
print("  Pro: Some caching within a column")
print("  Con: Still mostly misses")
print("")

print("Option 3: DON'T copy the TN, use self.tn directly")
print("  Pro: Environment sees updates automatically")
print("  Con: Must handle input tensors carefully")
print("")

print("Option 4: Accept stale environments as quasi-Newton")
print("  Pro: Maximum caching benefit")
print("  Con: Can cause instability (we're seeing this)")
print("")

print("Option 5: Clear cache only between EPOCHS")
print("  Pro: Reuse across all nodes within an epoch")
print("  Con: Very stale, but maybe OK with high regularization")
print("")

print("RECOMMENDATION: Option 1 or 3")
print("  Standard DMRG rebuilds env after each update (Option 1)")
print("  Or make the cache reference self.tn directly (Option 3)")
