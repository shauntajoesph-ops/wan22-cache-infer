"""First-Block Cache (FBCache) utilities.

This module defines the minimal state and helpers required to implement
First-Block Cache gating as specified in FB_CACHE.md. FBCache uses an early,
low-cost signal derived from the first transformer block to decide whether
to reuse the previous step's full-stack residual and skip the rest of the
transformer blocks for the current step.

Design requirements satisfied here:
- Inference-only; no training state or optimizers involved.
- Separate per-branch (cond/uncond) states for CFG pipelines; no leakage.
- Warmup/last-steps guards; lifecycle reset per run; counters for telemetry.
- Device/dtype/shape guards and fail-safe paths to avoid crashes.
- Distributed-friendly: the gating produces a single scalar suitable for
  all-reduce outside this module; we only expose scalar values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import torch


@dataclass
class FBCacheBranchState:
    """Per-branch state for FBCache (cond/uncond).

    Fields:
        prev_sig: Previous step's scalar signature (CPU float). The definition
                  depends on the chosen metric (hidden or residual based).
        prev_residual: Cached full-stack residual tensor (output - input) from
                  the last computed step; used when skipping.
        accum: Accumulator of rescaled metric values across steps; reset on
               compute, compared to threshold for gating.
        ema_val: Previous EMA value for the metric (if EMA > 0); None if unused.
        shape: Shape guard of the residual to prevent mismatched reuse.
        dtype: Dtype guard of the residual to prevent mismatched reuse.
        total/skipped/sum_rel/sum_rescaled/count_rel: Telemetry counters.
    """

    prev_sig: Optional[float] = None
    prev_residual: Optional[torch.Tensor] = None
    accum: float = 0.0
    ema_val: Optional[float] = None
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[torch.dtype] = None
    # Telemetry
    total: int = 0
    skipped: int = 0
    sum_rel: float = 0.0
    sum_rescaled: float = 0.0
    count_rel: int = 0


@dataclass
class FBCacheState:
    """Global FBCache state attached to a model instance.

    Fields:
        enabled: Whether FBCache gating is active.
        num_steps: Denosing steps in the current run.
        cnt: Executed step counter (increments on cond branch only in
             separate-CFG pipelines).
        warmup/last_steps: Guards to force compute early and late steps.
        thresh: Base threshold for the rescaled metric accumulator.
        metric: Metric identifier ('hidden_rel_l1' default, or 'residual_rel_l1').
        downsample: Stride for computing the metric (>=1); reduces overhead.
        ema: EMA factor in [0,1) for smoothing raw metric; 0 disables EMA.
        cfg_sep_diff: If True, compute CFG and non-CFG diffs separately; if
             False, reuse non-CFG diff for CFG decision.
        branch: Active branch label 'cond' or 'uncond'. Pipelines must set
             this before each forward.
        cond/uncond: Per-branch states.
        sp_world_size: For telemetry/sanity in SP; not used for math here.
        failsafe_count: Count of forced-compute events due to anomalies.
    """

    enabled: bool = False
    num_steps: int = 0
    cnt: int = 0
    warmup: int = 1
    last_steps: int = 1
    thresh: float = 0.08
    metric: str = "hidden_rel_l1"
    downsample: int = 1
    ema: float = 0.0
    cfg_sep_diff: bool = True
    branch: Literal["cond", "uncond"] = "cond"
    cond: FBCacheBranchState = field(default_factory=FBCacheBranchState)
    uncond: FBCacheBranchState = field(default_factory=FBCacheBranchState)
    sp_world_size: int = 1
    failsafe_count: int = 0
    # Store last computed cond diff for optional reuse on CFG step when
    # cfg_sep_diff is False. We store both raw and rescaled values to allow
    # consistent accumulation on the CFG branch.
    last_cond_rel: Optional[float] = None
    last_cond_rescaled: Optional[float] = None

    def branch_state(self) -> FBCacheBranchState:
        """Return the state for the currently active branch."""
        return self.cond if self.branch == "cond" else self.uncond


def summarize_hidden(x: torch.Tensor, downsample: int = 1) -> float:
    """Compute scalar signature from hidden states (mean absolute value).

    We optionally stride the last dimension of sequence tokens to reduce
    overhead. The resulting scalar is moved to CPU and returned as a Python
    float to ensure stable storage across device transitions.
    """
    if downsample > 1:
        # Stride across the token dimension (dimension 1) to reduce work.
        x = x[:, ::downsample]
    sig = x.abs().mean()
    return float(sig.detach().to("cpu"))


def summarize_residual(residual: torch.Tensor, downsample: int = 1) -> float:
    """Compute scalar signature from a residual tensor (mean absolute value)."""
    if downsample > 1:
        residual = residual[:, ::downsample]
    sig = residual.abs().mean()
    return float(sig.detach().to("cpu"))


def rescale_metric(rel: float, policy: str = "linear") -> float:
    """Rescale the raw relative metric according to the policy.

    Currently only 'linear' is supported; unknown policies fall back to
    identity to remain conservative.
    """
    if policy == "linear":
        return rel
    return rel


def move_residual_to(t: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Ensure the cached residual tensor is on the correct device/dtype.

    This helper is used before applying a cached residual during a skip
    decision to avoid device/dtype mismatches.
    """
    if t.device != device:
        t = t.to(device)
    if t.dtype != dtype:
        t = t.to(dtype)
    return t


def reset_branch(state: FBCacheBranchState) -> None:
    """Clear a single branch state to avoid stale reuse across runs."""
    state.prev_sig = None
    state.prev_residual = None
    state.accum = 0.0
    state.ema_val = None
    state.shape = None
    state.dtype = None
    state.total = 0
    state.skipped = 0
    state.sum_rel = 0.0
    state.sum_rescaled = 0.0
    state.count_rel = 0


def reset(state: FBCacheState) -> None:
    """Reset the full FBCache state at the beginning of a run."""
    state.cnt = 0
    reset_branch(state.cond)
    reset_branch(state.uncond)
