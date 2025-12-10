"""
EvoGrad utilities module.

This module provides utility functions and classes for:
    - Device management (CPU/CUDA/MPS auto-detection)
    - Tensor conversion and validation
    - Duplicate elimination in populations
    - Callbacks for monitoring optimisation
    - Seeding for reproducibility

Submodules:
    - device: Hardware device utilities
    - duplicates: Population duplicate detection and removal  
    - callbacks: Optimisation monitoring and control
"""

from evograd.utils.device import (
    # Device detection
    get_device,
    default_device,
    reset_default_device,
    get_default_dtype,
    # Tensor utilities
    ensure_tensor,
    ensure_bounds,
    to_device,
    # Reproducibility
    set_seed,
    # Memory and sync
    sync_device,
    get_memory_info,
    # Context manager
    DeviceContext,
    # Type alias
    TensorLike,
)

from evograd.utils.duplicates import (
    DuplicateMethod,
    DuplicateEliminator,
    eliminate_duplicates,
    has_duplicates,
    count_duplicates,
)

from evograd.utils.callbacks import (
    CallbackEvent,
    CallbackState,
    Callback,
    HistoryCallback,
    EarlyStoppingCallback,
    ConvergenceCallback,
    PrintCallback,
    CheckpointCallback,
    CompositeCallback,
    CallbackList,
    create_default_callbacks,
)

__all__ = [
    # Device detection
    "get_device",
    "default_device",
    "reset_default_device",
    "get_default_dtype",
    # Tensor utilities
    "ensure_tensor",
    "ensure_bounds",
    "to_device",
    # Reproducibility
    "set_seed",
    # Memory and sync
    "sync_device",
    "get_memory_info",
    # Context manager
    "DeviceContext",
    # Type alias
    "TensorLike",
    # Duplicate elimination
    "DuplicateMethod",
    "DuplicateEliminator",
    "eliminate_duplicates",
    "has_duplicates",
    "count_duplicates",
    # Callbacks
    "CallbackEvent",
    "CallbackState",
    "Callback",
    "HistoryCallback",
    "EarlyStoppingCallback",
    "ConvergenceCallback",
    "PrintCallback",
    "CheckpointCallback",
    "CompositeCallback",
    "CallbackList",
    "create_default_callbacks",
]
