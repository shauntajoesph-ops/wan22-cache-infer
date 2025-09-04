"""Shared cache-gated block runner.

This helper centralizes the CacheManager integration used by non‑SP, SP, and S2V
forward paths. It builds early signals, queries the manager, applies skip safely
or computes, and updates residuals. Callers provide small builder functions for
model‑family specifics (mod_inp construction, optional block‑0 residual, and
per‑block post transforms like S2V's after_transformer_block).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


BuildModInp = Callable[[torch.nn.Module, torch.Tensor, dict], torch.Tensor]
BuildBlock0Residual = Callable[[torch.nn.Module, torch.Tensor, dict], torch.Tensor]
PerBlockPost = Callable[[torch.nn.Module, int, torch.Tensor, dict], torch.Tensor]


def run_with_cache_manager(
    model: torch.nn.Module,
    x: torch.Tensor,
    kwargs: dict,
    build_mod_inp: BuildModInp,
    build_block0_residual: Optional[BuildBlock0Residual] = None,
    per_block_post: Optional[PerBlockPost] = None,
) -> torch.Tensor:
    """Run model blocks with unified CacheManager gating.

    - If no manager or all modes disabled, runs baseline compute (with optional per_block_post).
    - Else, builds early signals, asks manager, applies skip safely or computes, and updates.

    Args:
        model: Module with attributes `blocks` and optional `cache_manager`.
        x: Input hidden states to the block stack.
        kwargs: Forward kwargs passed to each block.
        build_mod_inp: Function that returns modulated first‑block input tensor.
        build_block0_residual: Optional function to compute block‑0 output when FB residual metric is active.
        per_block_post: Optional per‑block post transform; given (model, idx, x, kwargs) → x.

    Returns:
        Tensor after running the full block stack (or skip application).
    """
    cache_mgr = getattr(model, "cache_manager", None)
    fb_enabled = bool(getattr(getattr(cache_mgr, "cfg", object()), "enable_fb", False))
    tc_enabled = bool(getattr(getattr(cache_mgr, "cfg", object()), "enable_tc", False))

    if cache_mgr is None or (not fb_enabled and not tc_enabled):
        # Baseline: no caching enabled
        for idx, block in enumerate(model.blocks):
            x = block(x, **kwargs)
            if per_block_post is not None:
                x = per_block_post(model, idx, x, kwargs)
        return x

    # Build early signals for gating
    mod_inp = build_mod_inp(model, x, kwargs)
    x_after_block0 = None
    metric = str(getattr(cache_mgr.cfg, "fb_metric", "hidden_rel_l1")) if fb_enabled else ""
    if fb_enabled and metric.startswith("residual") and build_block0_residual is not None:
        x_after_block0 = build_block0_residual(model, x, kwargs)

    decision = cache_mgr.decide(x=x, mod_inp=mod_inp, x_after_block0=x_after_block0)
    x_before = x
    x, resume_from, applied_skip = cache_mgr.apply(decision, x)

    # Compute either when requested, or when a planned skip failed to apply (failsafe)
    if decision.action == "compute" or not applied_skip:
        if resume_from <= 0:
            for idx, block in enumerate(model.blocks):
                x = block(x, **kwargs)
                if per_block_post is not None:
                    x = per_block_post(model, idx, x, kwargs)
        else:
            # Resume after block‑0 when residual metric path is active
            if x_after_block0 is None:
                x = model.blocks[0](x, **kwargs)
                if per_block_post is not None:
                    x = per_block_post(model, 0, x, kwargs)
            else:
                x = x_after_block0
            for idx, block in enumerate(model.blocks[1:], start=1):
                x = block(x, **kwargs)
                if per_block_post is not None:
                    x = per_block_post(model, idx, x, kwargs)
        cache_mgr.update(decision, x_before=x_before, x_after=x)

    return x
