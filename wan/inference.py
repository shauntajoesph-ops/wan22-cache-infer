import os
from contextlib import contextmanager

import torch

INFERENCE_ONLY = os.getenv("WAN_INFERENCE_ONLY", "1") != "0"


def mark_inference_only(module: torch.nn.Module) -> torch.nn.Module:
    """
    Enforce inference-only behavior on a torch module.

    - Sets eval() and disables gradients on all parameters
    - Monkey-patches `.train()` to prevent switching back to training

    Returns the same module for convenience.
    """
    if module is None:
        return module
    module.eval().requires_grad_(False)

    def _disabled_train(self, mode: bool = True):  # pragma: no cover
        raise RuntimeError(
            "Training is disabled in this inference-only build. Use WAN_INFERENCE_ONLY=0 to allow training explicitly.")

    # Bind per-instance to avoid global side effects
    module.train = _disabled_train.__get__(module, module.__class__)  # type: ignore[attr-defined]
    return module


@contextmanager
def inference_guard():
    """Context manager to run with autograd disabled for inference."""
    with torch.inference_mode():
        yield


def enforce_on(*modules: torch.nn.Module) -> None:
    """Apply `mark_inference_only` to each non-None module."""
    for m in modules:
        mark_inference_only(m)

