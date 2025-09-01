"""TeaCache utility structures and helpers.

This module encapsulates the minimal state and helper functions required to
implement TeaCache-style conditional transformer skipping during inference.

Design constraints:
- Inference-only usage; no training code or optimizer state.
- Minimal, reversible integration behind flags; safe by default (disabled).
- Supports single-GPU and multi-GPU (SP/FSDP) with model offload.

Public dataclasses expose clear semantics and are documented for maintainers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import torch


@dataclass
class TeaCacheBranchState:
    """Per-branch (cond/uncond) TeaCache state.

    Stores the previous modulated-input signature, the cached residual tensor
    (i.e., output minus input across the DiT block stack), and an accumulator
    used to decide when to compute versus skip.

    Args:
        prev_mod_sig: Previous step's modulated input signature (CPU float). None if uninitialized.
        prev_residual: Cached residual tensor for skip path. May reside on CPU when offloaded.
        accum: Accumulated, rescaled relative change value used for gating.
        shape: Shape of the tensor to which residual applies (guard against mismatch across runs or shapes).
        dtype: Dtype of the residual tensor (guard against mismatch across dtype changes).
    """

    prev_mod_sig: Optional[float] = None  # previous step signature (CPU scalar)
    prev_residual: Optional[torch.Tensor] = None  # cached residual tensor
    accum: float = 0.0  # accumulator for rescaled rel-changes
    shape: Optional[Tuple[int, ...]] = None  # guard: expected residual shape
    dtype: Optional[torch.dtype] = None  # guard: expected residual dtype
    # Telemetry
    total: int = 0  # number of gating decisions on this branch
    skipped: int = 0  # number of times skip path was taken
    sum_rel: float = 0.0  # sum of raw rel values observed (for averaging)
    sum_rescaled: float = 0.0  # sum of rescaled values
    count_rel: int = 0  # how many rel/rescaled samples accumulated


@dataclass
class TeaCacheState:
    """Global TeaCache state attached to a DiT model instance.

    Aggregates per-branch state and run-level configuration.

    Args:
        enabled: Whether TeaCache gating is enabled for this model.
        num_steps: Number of denoising steps for this run.
        cnt: Current step index (increments once per step on the 'cond' branch only).
        warmup: Number of initial steps to force compute (no skipping).
        last_steps: Number of final steps to force compute (no skipping).
        thresh: Accumulator threshold below which we skip (compute when >= thresh).
        policy: Rescale policy identifier. 'linear' applies identity mapping.
        branch: Active CFG branch: 'cond' or 'uncond'. Pipelines must set this before each call.
        cond: Per-branch state for the conditional forward.
        uncond: Per-branch state for the unconditional forward.
        run_id: Monotonic counter per run; prevents stale state reuse across runs.
        sp_world_size: World size in sequence-parallel mode; used for sanity checks/telemetry.
    """

    enabled: bool = False
    num_steps: int = 0
    cnt: int = 0
    warmup: int = 1
    last_steps: int = 1
    thresh: float = 0.08
    policy: str = "linear"
    branch: Literal["cond", "uncond"] = "cond"
    cond: TeaCacheBranchState = field(default_factory=TeaCacheBranchState)
    uncond: TeaCacheBranchState = field(default_factory=TeaCacheBranchState)
    run_id: int = 0
    sp_world_size: int = 1
    # Telemetry
    failsafe_count: int = 0  # number of times we forced compute due to a guard

    def branch_state(self) -> TeaCacheBranchState:
        """Return the active branch state based on `branch`.

        Returns:
            TeaCacheBranchState: The state object for the active branch.
        """

        return self.cond if self.branch == "cond" else self.uncond


def summarize_mod(x: torch.Tensor) -> float:
    """Compute a compact signature of the first-block modulated input.

    We use a scalar mean absolute value which is simple and effective as a
    temporal-change proxy. This is computed on the current device then moved to
    CPU as a Python float for stable storage across device/offload transitions.

    Args:
        x: The modulated input tensor of shape [B, L, C].

    Returns:
        float: Mean absolute value as a Python float.
    """

    # Take mean of absolute values to obtain a scale-invariant, robust signal.
    sig = x.abs().mean()
    # Move to CPU and convert to Python float for storage in state.
    return float(sig.detach().to("cpu"))


def rescale(rel: float, policy: str = "linear") -> float:
    """Rescale the raw relative change according to the selected policy.

    The default 'linear' policy applies identity mapping, which is conservative
    and model-agnostic. Polynomial policies can be added later if calibrated
    per model family (e.g., FLUX/Mochi-style).

    Args:
        rel: Raw relative change value (non-negative).
        policy: Policy identifier. Currently supports only 'linear'.

    Returns:
        float: Rescaled value used in the accumulator.
    """

    if policy == "linear":
        return rel
    # Unknown policies fall back to identity to remain safe.
    return rel


def move_residual_to(t: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Move or cast the cached residual tensor to the target device/dtype.

    Args:
        t: Cached residual tensor.
        device: Target device (usually the model's device for current forward).
        dtype: Target dtype (model's compute dtype, e.g., bf16/fp16/fp32).

    Returns:
        torch.Tensor: Tensor located on `device` with dtype `dtype`.
    """

    if t.device != device:
        t = t.to(device)
    if t.dtype != dtype:
        t = t.to(dtype)
    return t


def reset_branch(state: TeaCacheBranchState) -> None:
    """Reset a branch state to its uninitialized form.

    Clears accumulator, signature, residual, and guards to avoid stale reuse
    across runs or shape/dtype changes.

    Args:
        state: Branch state to reset.
    """

    state.prev_mod_sig = None
    state.prev_residual = None
    state.accum = 0.0
    state.shape = None
    state.dtype = None


def reset(state: TeaCacheState) -> None:
    """Reset the full TeaCache state across both branches.

    Args:
        state: Global TeaCache state attached to a model.
    """

    state.cnt = 0
    reset_branch(state.cond)
    reset_branch(state.uncond)
