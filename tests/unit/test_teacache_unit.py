import math
import torch

from wan.utils.teacache import (
    TeaCacheState,
    TeaCacheBranchState,
    summarize_mod,
    rescale,
    reset as tc_reset,
    move_residual_to as tc_move_residual_to,
)


def test_summarize_mod_returns_python_float():
    x = torch.tensor([[[1.0, -2.0], [0.0, 3.0]]])  # [1,2,2]
    val = summarize_mod(x)
    assert isinstance(val, float)
    assert math.isfinite(val)


def test_rescale_linear_identity_and_fallback():
    for v in [0.0, 0.05, 1.23]:
        assert rescale(v, "linear") == v
        assert rescale(v, "unknown") == v


def test_move_residual_to_dtype_device_safety_cpu():
    t = torch.ones(3, 4, dtype=torch.float32)
    t2 = tc_move_residual_to(t, t.device, torch.float16)
    assert t2.dtype == torch.float16
    assert t2.device == t.device


def test_state_reset_branch_independence():
    st = TeaCacheState(enabled=True, num_steps=10, thresh=0.08)
    # Simulate usage on cond branch
    st.branch = 'cond'
    st.cond.prev_mod_sig = 0.1
    st.cond.accum = 0.02
    st.cond.total = 2
    # And uncond branch with different values
    st.branch = 'uncond'
    st.uncond.prev_mod_sig = 0.3
    st.uncond.accum = 0.04
    st.uncond.total = 3

    tc_reset(st)
    assert st.cnt == 0
    assert st.cond.prev_mod_sig is None and st.uncond.prev_mod_sig is None
    assert st.cond.accum == 0.0 and st.uncond.accum == 0.0
    assert st.cond.total == 0 and st.uncond.total == 0


def test_warmup_last_step_flags_present():
    st = TeaCacheState(warmup=2, last_steps=2, num_steps=10)
    assert st.warmup == 2 and st.last_steps == 2

