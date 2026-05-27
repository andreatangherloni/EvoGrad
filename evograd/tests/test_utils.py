"""
Test script for EvoGrad utils module.

Tests:
    - device.py: Device detection and tensor movement
    - duplicates.py: Duplicate elimination
    - callbacks.py: Callback system

Usage:
    python -m evograd.tests.test_utils
"""

import sys
import torch
import torch.nn as nn
import tempfile
import os

# Add parent of evograd to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evograd.utils.device import get_device, to_device
from evograd.utils.duplicates import (
    DuplicateMethod,
    DuplicateEliminator,
    eliminate_duplicates,
    has_duplicates,
    count_duplicates,
)
from evograd.utils.callbacks import (
    Callback,
    CallbackList,
    CallbackState,
    HistoryCallback,
    PrintCallback,
    EarlyStoppingCallback,
    CheckpointCallback,
    ConvergenceCallback,
)


def test_device():
    """Test device detection and tensor movement."""
    print("\n" + "="*60)
    print("Testing device.py")
    print("="*60)
    
    # Test get_device
    print("\n1. Testing get_device()...")
    device = get_device()
    print(f"   Default device: {device}")
    assert isinstance(device, torch.device)
    
    # Test with specific device string
    cpu_device = get_device("cpu")
    print(f"   CPU device: {cpu_device}")
    assert cpu_device.type == "cpu"
    
    # Test auto detection (None means auto)
    auto_device = get_device(None)
    print(f"   Auto device: {auto_device}")
    
    # Test to_device with tensors (keyword-only device argument)
    print("\n2. Testing to_device() with tensors...")
    x = torch.randn(10, 5)
    y = torch.randn(3, 3)
    x_moved, y_moved = to_device(x, y, device=cpu_device)
    print(f"   Moved tensors to {x_moved.device}")
    assert x_moved.device == cpu_device
    assert y_moved.device == cpu_device
    
    # Test to_device with single tensor
    print("\n3. Testing to_device() with single tensor...")
    z = torch.randn(5)
    (z_moved,) = to_device(z, device=cpu_device)
    print(f"   Moved single tensor to {z_moved.device}")
    assert z_moved.device == cpu_device
    
    print("\n✓ device.py tests passed!")


def test_duplicates():
    """Test duplicate elimination strategies."""
    print("\n" + "="*60)
    print("Testing duplicates.py")
    print("="*60)
    
    # Create test population with duplicates
    pop = torch.tensor([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],  # Exact duplicate of row 0
        [4.0, 5.0, 6.0],
        [1.0001, 2.0001, 3.0001],  # Near duplicate of row 0
        [7.0, 8.0, 9.0],
    ])
    
    xl = torch.zeros(3)
    xu = torch.ones(3) * 10
    
    # Test DuplicateEliminator with EPSILON_L2
    print("\n1. Testing DuplicateEliminator (EPSILON_L2)...")
    eliminator = DuplicateEliminator(
        method=DuplicateMethod.EPSILON_L2,
        epsilon=0.01,
    )
    
    # Eliminate duplicates by calling the eliminator
    new_pop = eliminator(pop, xl, xu)
    print(f"   Original population shape: {pop.shape}")
    print(f"   Population after elimination: {new_pop.shape}")
    print(f"   Duplicates found: {eliminator.n_duplicates_found}")
    print(f"   Duplicates resolved: {eliminator.n_duplicates_resolved}")
    
    # Test has_duplicates function
    print("\n2. Testing has_duplicates()...")
    has_dups = has_duplicates(pop, epsilon=0.01)
    print(f"   Has duplicates: {has_dups}")
    assert has_dups == True, "Population should have duplicates"
    
    # Test count_duplicates function
    print("\n3. Testing count_duplicates()...")
    n_dups = count_duplicates(pop, epsilon=0.01)
    print(f"   Number of duplicates: {n_dups}")
    assert n_dups >= 1, "Should find at least 1 duplicate"
    
    # Test DuplicateEliminator with HASH method
    print("\n4. Testing DuplicateEliminator (HASH)...")
    hash_eliminator = DuplicateEliminator(
        method=DuplicateMethod.HASH,
        decimals=2,
    )
    new_pop_hash = hash_eliminator(pop, xl, xu)
    print(f"   Hash method duplicates found: {hash_eliminator.n_duplicates_found}")
    
    # Test DuplicateEliminator with NONE method
    print("\n5. Testing DuplicateEliminator (NONE)...")
    no_elim = DuplicateEliminator(method=DuplicateMethod.NONE)
    new_pop_none = no_elim(pop, xl, xu)
    assert torch.allclose(new_pop_none, pop), "NONE method should not modify population"
    print("   NONE method correctly leaves population unchanged")
    
    # Test eliminate_duplicates convenience function
    print("\n6. Testing eliminate_duplicates()...")
    cleaned_pop = eliminate_duplicates(pop, xl, xu, epsilon=0.01)
    print(f"   Cleaned population shape: {cleaned_pop.shape}")
    
    print("\n✓ duplicates.py tests passed!")


def test_callbacks():
    """Test callback system."""
    print("\n" + "="*60)
    print("Testing callbacks.py")
    print("="*60)
    
    # Test HistoryCallback
    print("\n1. Testing HistoryCallback...")
    history_cb = HistoryCallback(
        track_population=True,
        track_hyperparams=True,
        track_diversity=False,
        track_fitness_stats=True,
    )
    
    # Create initial state
    state = CallbackState(
        generation=0,
        n_evals=0,
        best_fitness=float('inf'),
        best_solution=torch.randn(5),
        current_fitness=torch.randn(10),
        current_population=torch.randn(10, 5),
    )
    
    history_cb.on_optimisation_start(state)
    
    for gen in range(5):
        state.generation = gen
        state.n_evals = (gen + 1) * 10
        state.best_fitness = 100.0 / (gen + 1)
        state.current_fitness = torch.randn(10)
        history_cb.on_generation_end(state)
    
    print(f"   Tracked generations: {len(history_cb.generations)}")
    print(f"   Best fitness history: {history_cb.best_fitness[:3]}...")
    # on_optimisation_start records the initial state, then each of the 5
    # on_generation_end calls records one more -> 1 + 5 = 6 entries.
    assert len(history_cb.generations) == 6
    assert len(history_cb.best_fitness) == 6
    
    # Test PrintCallback
    print("\n2. Testing PrintCallback...")
    print_cb = PrintCallback(every=2, show_time=True)
    
    state = CallbackState(generation=0, n_evals=0, best_fitness=100.0)
    print_cb.on_optimisation_start(state)
    
    for gen in range(4):
        state.generation = gen
        state.best_fitness = 100.0 - gen * 10
        state.n_evals = gen * 10
        print_cb.on_generation_end(state)
    
    print_cb.on_optimisation_end(state)
    print("   PrintCallback executed without errors")
    
    # Test EarlyStoppingCallback
    print("\n3. Testing EarlyStoppingCallback...")
    early_stop_cb = EarlyStoppingCallback(
        patience=3,
        min_delta=0.1,
    )
    
    state = CallbackState(generation=0, n_evals=0, best_fitness=100.0)
    early_stop_cb.on_optimisation_start(state)
    
    # Simulate improvement then stagnation
    fitness_sequence = [100, 90, 80, 80, 80, 80]  # Stagnates after 3rd
    stopped_at = None
    for gen, fit in enumerate(fitness_sequence):
        state.generation = gen
        state.best_fitness = fit
        early_stop_cb.on_generation_end(state)
        if state.stop_optimisation:
            stopped_at = gen
            break
    
    print(f"   Early stopping triggered at generation: {stopped_at}")
    assert stopped_at is not None, "Early stopping should have triggered"
    
    # Test CheckpointCallback
    print("\n4. Testing CheckpointCallback...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock module for checkpointing
        mock_module = nn.Linear(5, 1)
        
        checkpoint_cb = CheckpointCallback(
            directory=tmpdir,
            every=2,
            save_best_only=True,
        )
        
        state = CallbackState(
            generation=0,
            n_evals=0,
            best_fitness=100.0,
            algorithm=mock_module,
        )
        checkpoint_cb.on_optimisation_start(state)
        
        for gen in range(5):
            state.generation = gen
            state.best_fitness = 100.0 - gen * 20
            checkpoint_cb.on_generation_end(state)
        
        checkpoint_cb.on_optimisation_end(state)
        
        # Check files were created
        files = os.listdir(tmpdir)
        print(f"   Checkpoint files created: {files}")
        assert len(files) > 0, "Should have created checkpoint files"
    
    # Test ConvergenceCallback
    print("\n5. Testing ConvergenceCallback...")
    conv_cb = ConvergenceCallback(
        threshold=0.001,
        window=3,
        min_generations=0,
    )
    
    state = CallbackState(generation=0, n_evals=0, best_fitness=100.0)
    conv_cb.on_optimisation_start(state)
    
    # Simulate convergence
    fitness_sequence = [100.0, 50.0, 25.0, 24.999, 24.998, 24.997]
    stopped_at = None
    for gen, fit in enumerate(fitness_sequence):
        state.generation = gen
        state.best_fitness = fit
        conv_cb.on_generation_end(state)
        if state.stop_optimisation:
            stopped_at = gen
            break
    
    print(f"   Convergence detected at generation: {stopped_at}")
    
    # Test CallbackList
    print("\n6. Testing CallbackList...")
    cb_list = CallbackList([
        HistoryCallback(),
        PrintCallback(every=10),
    ])
    
    state = CallbackState(generation=0, n_evals=0, best_fitness=100.0)
    cb_list.on_optimisation_start(state)
    for gen in range(3):
        state.generation = gen
        state.best_fitness = 100.0 - gen
        cb_list.on_generation_end(state)
    cb_list.on_optimisation_end(state)
    print("   CallbackList executed all callbacks")
    
    print("\n✓ callbacks.py tests passed!")


def run_all_tests():
    """Run all utils tests."""
    print("\n" + "#"*60)
    print("# EvoGrad Utils Module Tests")
    print("#"*60)
    
    try:
        test_device()
        test_duplicates()
        test_callbacks()
        
        print("\n" + "="*60)
        print("✓ ALL UTILS TESTS PASSED!")
        print("="*60)
        return True
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
