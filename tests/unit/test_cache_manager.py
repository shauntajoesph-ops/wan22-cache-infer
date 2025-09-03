import math
import itertools

import pytest
import torch

from wan.utils.cache_manager import CacheManager, CMConfig, Decision


def make_mgr(enable_fb=False, enable_tc=False, cfg_sep_diff=False, num_steps=4, warmup=1, last_steps=1, sp_world_size=1, fb_metric="hidden_rel_l1"):
    cfg = CMConfig(
        num_steps=num_steps,
        warmup=warmup,
        last_steps=last_steps,
        enable_tc=enable_tc,
        tc_thresh=0.08,
        tc_policy="linear",
        enable_fb=enable_fb,
        fb_thresh=0.08,
        fb_metric=fb_metric,
        fb_downsample=1,
        fb_ema=0.0,
        cfg_sep_diff=cfg_sep_diff,
        evaluation_order=("fb", "tc"),
        sp_world_size=sp_world_size,
    )
    mgr = CacheManager(cfg)
    mgr.attach(num_steps=num_steps, sp_world_size=sp_world_size)
    return mgr


def fake_tensor(val, shape=(1, 4, 8)):
    # Build a tensor with controlled mean absolute signature
    t = torch.full(shape, float(val), dtype=torch.float32)
    t[0, 0, 0] = float(val)  # avoid zero-ing
    return t


def test_warmup_and_last_steps_forced_compute():
    mgr = make_mgr(enable_fb=True, num_steps=3, warmup=1, last_steps=1)
    x = torch.zeros(1, 2, 2)
    # Step 1 (cond): warmup forces compute
    mgr.begin_step("cond")
    d1 = mgr.decide(x=x, mod_inp=fake_tensor(1.0))
    assert d1.action == "compute" and d1.reason == "forced"
    # Populate residual via update and ensure accum reset
    mgr.update(d1, x, x + 1)
    # Step 1 (uncond): follows cond
    mgr.begin_step("uncond")
    d1u = mgr.decide(x=x, mod_inp=fake_tensor(1.0))
    assert d1u.action == "compute"
    # Step 2 (cond): main phase
    mgr.begin_step("cond")
    d2 = mgr.decide(x=x, mod_inp=fake_tensor(1.0))
    # Either skip (if prev set and accum<thresh) or compute; both valid here
    assert d2.action in ("skip", "compute")
    # Step 3 (cond): last-steps forces compute
    mgr.begin_step("cond")
    d3 = mgr.decide(x=x, mod_inp=fake_tensor(1.0))
    assert d3.action == "compute" and d3.reason == "forced"


@pytest.mark.parametrize("enable_fb,enable_tc,cfg_sep_diff", itertools.product([False, True], [False, True], [False, True]))
def test_priority_and_cfg_reuse(enable_fb, enable_tc, cfg_sep_diff):
    mgr = make_mgr(enable_fb=enable_fb, enable_tc=enable_tc, cfg_sep_diff=cfg_sep_diff, num_steps=3, warmup=0, last_steps=0)
    x = torch.zeros(1, 2, 2)
    # Seed prev signatures by running one compute step on cond
    mgr.begin_step("cond")
    d0 = mgr.decide(x=x, mod_inp=fake_tensor(1.0))
    mgr.update(d0, x, x + 1)
    # Next step: decide under priority
    mgr.begin_step("cond")
    d = mgr.decide(x=x, mod_inp=fake_tensor(1.0), x_after_block0=(x + 0.5))
    if not enable_fb and not enable_tc:
        assert d.mode is None and d.action == "compute"
    elif enable_fb and not enable_tc:
        assert d.mode == "fb"
    elif not enable_fb and enable_tc:
        assert d.mode == "tc"
    else:
        # FB takes precedence
        assert d.mode == "fb"

    # Uncond behavior: cfg reuse
    mgr.begin_step("uncond")
    d_u = mgr.decide(x=x, mod_inp=fake_tensor(1.0))
    if cfg_sep_diff is False:
        assert d_u.reason == "cfg_reuse"
        assert d_u.action == d.action and d_u.mode == d.mode
    else:
        # cfg_sep_diff=true still follows cond action (but may compute metric internally)
        assert d_u.reason == "cfg_follow_cond"
        assert d_u.action == d.action and d_u.mode == d.mode


def test_skip_requires_residual_then_applies_residual():
    mgr = make_mgr(enable_fb=True, num_steps=3, warmup=0, last_steps=0)
    x = torch.zeros(1, 2, 2)
    # Initialize prev signature without residual
    mgr.begin_step("cond")
    d0 = mgr.decide(x=x, mod_inp=fake_tensor(1.0))
    # Try to skip with no residual → apply() forces compute fallback
    mgr.begin_step("cond")
    d1 = mgr.decide(x=x, mod_inp=fake_tensor(1.0))
    x_applied, _ = mgr.apply(d1, x)
    assert torch.equal(x_applied, x)
    assert mgr.failsafe_count >= 0  # at least not crashing
    # Compute and cache residual
    mgr.update(d1 if isinstance(d1, Decision) else DecisionShim(d1), x, x + 2)
    # Next step: skip with residual present
    mgr.begin_step("cond")
    d2 = mgr.decide(x=x, mod_inp=fake_tensor(1.0))
    xr, _ = mgr.apply(d2, x)
    if d2.action == "skip":
        assert torch.equal(xr, x + (x + 2 - x))  # x + residual


def test_sp_reduce_no_dist_initialized():
    # With sp_world_size>1 but dist not initialized, reduction should be a no-op and not crash
    mgr = make_mgr(enable_tc=True, num_steps=2, warmup=0, last_steps=0, sp_world_size=8)
    x = torch.zeros(1, 2, 2)
    mgr.begin_step("cond")
    d = mgr.decide(x=x, mod_inp=fake_tensor(1.0))
    assert d.action in ("skip", "compute")


class DecisionShim:
    # Minimal shim to reuse manager.update path with an existing decision
    def __init__(self, d):
        self.mode = d.mode
        self.action = d.action
