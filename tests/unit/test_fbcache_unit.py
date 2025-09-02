import math
import os

import torch

from wan.utils.fbcache import (
    FBCacheBranchState,
    FBCacheState,
    move_residual_to,
    reset as fb_reset,
    summarize_hidden,
    summarize_residual,
    rescale_metric,
)


def test_summarize_hidden_and_residual_downsample():
    # Construct a simple hidden tensor with known mean absolute value
    x = torch.tensor([[[-1.0, 0.0, 1.0, 3.0], [2.0, -2.0, 0.0, 0.0]]])  # [B=1, L=2, C=4]
    # mean abs over all = (1+0+1+3 + 2+2+0+0)/8 = 9/8 = 1.125
    sig_full = summarize_hidden(x, downsample=1)
    assert abs(sig_full - 1.125) < 1e-6
    # Downsample by 2 along token dim: take positions 0 only for L=2 → same sample for this x
    sig_ds2 = summarize_hidden(x, downsample=2)
    # The mean over selected tokens differs due to reduced samples; verify it's finite and non-negative
    assert math.isfinite(sig_ds2) and sig_ds2 >= 0.0

    # Residual summarization mirrors hidden summarization
    r = x.clone()
    sig_r = summarize_residual(r, downsample=1)
    assert abs(sig_r - sig_full) < 1e-6


def test_rescale_metric_linear_identity():
    # Linear policy is identity and should be conservative
    vals = [0.0, 0.1, 1.0]
    for v in vals:
        assert rescale_metric(v, "linear") == v
        # Unknown policy falls back to identity
        assert rescale_metric(v, "unknown") == v


def test_move_residual_to_dtype_changes_cpu_only():
    # Ensure dtype cast works even without CUDA availability
    t = torch.ones(2, 3, dtype=torch.float32)
    t2 = move_residual_to(t, t.device, torch.float16)
    assert t2.dtype == torch.float16
    # Device unchanged in CPU-only environment
    assert t2.device == t.device


def test_state_reset_and_accumulator():
    st = FBCacheState(enabled=True, num_steps=10, thresh=0.08)
    # Simulate accumulation
    br = st.branch_state()
    br.accum = 0.05
    br.total = 3
    br.skipped = 2
    fb_reset(st)
    # After reset, counters and accumulators cleared
    assert st.cnt == 0
    assert st.cond.accum == 0.0 and st.uncond.accum == 0.0
    assert st.cond.total == 0 and st.uncond.total == 0
    assert st.cond.prev_sig is None and st.uncond.prev_sig is None


def test_cfg_sep_diff_fields_present():
    # Ensure fields supporting CFG reuse are present and mutable
    st = FBCacheState(cfg_sep_diff=False)
    st.last_cond_rel = 0.01
    st.last_cond_rescaled = 0.01
    assert st.last_cond_rel == st.last_cond_rescaled == 0.01

