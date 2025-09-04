import torch
import pytest

from wan.utils.cache_manager import CacheManager, CMConfig
from wan.utils.cache_runner import run_with_cache_manager


class AddConstBlock(torch.nn.Module):
    def __init__(self, c: float):
        super().__init__()
        self.c = c

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x + self.c

    __call__ = forward


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Single block that adds a constant so we can detect compute
        self.blocks = torch.nn.ModuleList([AddConstBlock(1.0)])
        self.cache_manager = None


def build_mod_inp(_m: torch.nn.Module, x: torch.Tensor, _kw: dict) -> torch.Tensor:
    # Constant signal across steps to drive TeaCache skip when prev exists
    return torch.zeros_like(x)


def test_cfg_reuse_uncond_missing_residual_forces_compute():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = torch.zeros(2, 3, device=device)
    m = FakeModel().to(device)

    cfg = CMConfig(
        num_steps=2,
        warmup=0,
        last_steps=0,
        enable_tc=True,
        tc_thresh=0.08,
        enable_fb=False,
        evaluation_order=("fb", "tc"),
    )
    mgr = CacheManager(cfg)
    mgr.attach(num_steps=2)
    m.cache_manager = mgr

    # Step 1 cond: compute (prev=None)
    mgr.begin_step("cond")
    out1c = run_with_cache_manager(
        model=m, x=x0, kwargs={}, build_mod_inp=build_mod_inp
    )
    # Step 1 uncond: compute
    mgr.begin_step("uncond")
    out1u = run_with_cache_manager(
        model=m, x=x0, kwargs={}, build_mod_inp=build_mod_inp
    )
    assert torch.allclose(out1c, x0 + 1)
    assert torch.allclose(out1u, x0 + 1)

    # Step 2 cond: identical signal, so TeaCache proposes skip; cond residual exists so skip applies
    mgr.begin_step("cond")
    out2c = run_with_cache_manager(
        model=m, x=x0, kwargs={}, build_mod_inp=build_mod_inp
    )
    # No compute => output remains x0 (skip applied)
    assert torch.allclose(out2c, x0)

    # Step 2 uncond: drop residual to simulate divergence; CFG reuse would propose skip,
    # but manager must force compute due to missing residual in uncond branch.
    mgr.uncond.residual = None
    mgr.uncond.shape = None
    mgr.uncond.dtype = None
    prev_div = mgr.pair_divergence_failsafes
    mgr.begin_step("uncond")
    out2u = run_with_cache_manager(
        model=m, x=x0, kwargs={}, build_mod_inp=build_mod_inp
    )
    # Compute happened => output equals x0 + 1
    assert torch.allclose(out2u, x0 + 1)
    assert mgr.pair_divergence_failsafes == prev_div + 1


def test_apply_returns_fallback_when_skip_unavailable():
    # Directly validate apply() signals fallback when residual is missing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = torch.zeros(2, 3, device=device)
    m = FakeModel().to(device)

    cfg = CMConfig(
        num_steps=2,
        warmup=0,
        last_steps=0,
        enable_tc=True,
        tc_thresh=0.08,
        enable_fb=False,
        evaluation_order=("fb", "tc"),
    )
    mgr = CacheManager(cfg)
    mgr.attach(num_steps=2)
    m.cache_manager = mgr

    # Prime prev signature and residual via one compute
    mgr.begin_step("cond")
    _ = run_with_cache_manager(
        model=m, x=x0, kwargs={}, build_mod_inp=build_mod_inp
    )

    # Next cond step should decide skip; emulate decision/apply directly
    mgr.begin_step("cond")
    d = mgr.decide(x=x0, mod_inp=build_mod_inp(m, x0, {}), x_after_block0=None)
    assert d.action == "skip"
    # Drop residual to force apply failsafe
    mgr.cond.residual = None
    mgr.cond.shape = None
    mgr.cond.dtype = None
    x_out, resume_from, applied = mgr.apply(d, x0)
    assert not applied
    assert torch.allclose(x_out, x0)
    assert isinstance(resume_from, int)


def test_cond_skip_apply_failsafe_triggers_compute_via_runner(monkeypatch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = torch.zeros(2, 3, device=device)
    m = FakeModel().to(device)

    cfg = CMConfig(
        num_steps=2,
        warmup=0,
        last_steps=0,
        enable_tc=True,
        tc_thresh=0.08,
        enable_fb=False,
        evaluation_order=("fb", "tc"),
    )
    mgr = CacheManager(cfg)
    mgr.attach(num_steps=2)
    m.cache_manager = mgr

    # Step 1 cond: compute (prime residual)
    mgr.begin_step("cond")
    _ = run_with_cache_manager(model=m, x=x0, kwargs={}, build_mod_inp=build_mod_inp)

    # Step 2 cond: force residual drop right before apply() to trigger fallback
    orig_apply = mgr.apply

    def apply_and_drop(decision, x):
        mgr.cond.residual = None
        mgr.cond.shape = None
        mgr.cond.dtype = None
        return orig_apply(decision, x)

    monkeypatch.setattr(mgr, "apply", apply_and_drop)
    mgr.begin_step("cond")
    out = run_with_cache_manager(model=m, x=x0, kwargs={}, build_mod_inp=build_mod_inp)
    # Since skip failed, compute must have happened => +1
    assert torch.allclose(out, x0 + 1)


def test_cfg_sep_diff_true_uncond_missing_residual_computes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = torch.zeros(2, 3, device=device)
    m = FakeModel().to(device)

    cfg = CMConfig(
        num_steps=2,
        warmup=0,
        last_steps=0,
        enable_tc=True,
        tc_thresh=0.08,
        enable_fb=False,
        cfg_sep_diff=True,
        evaluation_order=("fb", "tc"),
    )
    mgr = CacheManager(cfg)
    mgr.attach(num_steps=2)
    m.cache_manager = mgr

    # Step 1 cond/uncond compute
    mgr.begin_step("cond"); _ = run_with_cache_manager(model=m, x=x0, kwargs={}, build_mod_inp=build_mod_inp)
    mgr.begin_step("uncond"); _ = run_with_cache_manager(model=m, x=x0, kwargs={}, build_mod_inp=build_mod_inp)

    # Step 2 cond: skip applies, but for uncond residual is missing => apply fails -> compute
    mgr.begin_step("cond"); _ = run_with_cache_manager(model=m, x=x0, kwargs={}, build_mod_inp=build_mod_inp)
    mgr.uncond.residual = None; mgr.uncond.shape = None; mgr.uncond.dtype = None
    mgr.begin_step("uncond")
    out_u = run_with_cache_manager(model=m, x=x0, kwargs={}, build_mod_inp=build_mod_inp)
    assert torch.allclose(out_u, x0 + 1)


def test_sp_reduction_local_and_initialized(tmp_path):
    # local: dist not initialized
    device = torch.device("cpu")
    x0 = torch.zeros(1, 2, device=device)
    m = FakeModel().to(device)
    cfg = CMConfig(num_steps=2, warmup=0, last_steps=0, enable_tc=True)
    mgr = CacheManager(cfg); mgr.attach(num_steps=2); m.cache_manager = mgr
    mgr.begin_step("cond"); _ = run_with_cache_manager(model=m, x=x0, kwargs={}, build_mod_inp=build_mod_inp)
    mgr.begin_step("cond"); d = mgr.decide(x=x0, mod_inp=build_mod_inp(m, x0, {}), x_after_block0=None)
    assert d.action in ("compute", "skip")

    # initialized: single-process gloo
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
    init_file = tmp_path / "pg"
    dist.init_process_group(backend="gloo", init_method=f"file://{init_file}", rank=0, world_size=1)
    try:
        mgr2 = CacheManager(cfg); mgr2.attach(num_steps=2); m.cache_manager = mgr2
        mgr2.begin_step("cond"); _ = run_with_cache_manager(model=m, x=x0, kwargs={}, build_mod_inp=build_mod_inp)
        mgr2.begin_step("cond"); d2 = mgr2.decide(x=x0, mod_inp=build_mod_inp(m, x0, {}), x_after_block0=None)
        assert d2.action in ("compute", "skip")
        # Summary should not crash
        _ = mgr2.summary()
    finally:
        dist.destroy_process_group()


def test_offload_transitions_and_dtype_shape_guards():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = torch.zeros(2, 3, device=device)
    m = FakeModel().to(device)
    cfg = CMConfig(num_steps=3, warmup=0, last_steps=0, enable_tc=True)
    mgr = CacheManager(cfg); mgr.attach(num_steps=3); m.cache_manager = mgr

    # Prime residual on current device (compute)
    mgr.begin_step("cond"); _ = run_with_cache_manager(model=m, x=x0, kwargs={}, build_mod_inp=build_mod_inp)
    assert mgr.cond.residual is not None

    # Move to CPU then back (if CUDA available)
    mgr.move_cached_residuals_to(torch.device("cpu"))
    assert mgr.cond.residual.device.type == "cpu"
    if torch.cuda.is_available():
        mgr.move_cached_residuals_to(torch.device("cuda"))
        assert mgr.cond.residual.device.type == "cuda"

    # Dtype guard: make x half while residual is float32 → decision should compute due to residual_guard
    x_half = x0.half()
    mgr.begin_step("cond")
    d = mgr.decide(x=x_half, mod_inp=build_mod_inp(m, x_half, {}), x_after_block0=None)
    assert d.action == "compute"
    assert d.reason in ("residual_guard", "forced", "tc>=thresh")

    # Shape guard: tamper recorded shape to mismatch and ensure compute
    mgr.cond.shape = (999,)
    d2 = mgr.decide(x=x0, mod_inp=build_mod_inp(m, x0, {}), x_after_block0=None)
    assert d2.action == "compute"


def test_warmup_and_last_steps_per_expert_behavior():
    # Two managers (simulate I2V low/high experts) share the same num_steps and enforce warmup/last_steps
    cfg = CMConfig(num_steps=4, warmup=1, last_steps=1, enable_tc=True)
    mgr_low = CacheManager(cfg); mgr_low.attach(num_steps=4)
    mgr_high = CacheManager(cfg); mgr_high.attach(num_steps=4)
    x = torch.zeros(1, 2)

    def step_and_check(mgr, step_idx):
        mgr.begin_step("cond")
        d = mgr.decide(x=x, mod_inp=torch.zeros_like(x), x_after_block0=None)
        if step_idx in (1, 4):
            assert d.reason == "forced" and d.action == "compute"
        return d

    # Run 4 steps for both experts
    for i in range(1, 5):
        _ = step_and_check(mgr_low, i)
        _ = step_and_check(mgr_high, i)


def test_s2v_residual_metric_resume_and_skip():
    # Simulate S2V-like flow with residual metric and ensure block-0 is not recomputed on resume
    class Block0Identity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            self.calls += 1
            return x  # identity so residual metric is zero/stable

    class Block1Add(torch.nn.Module):
        def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            return x + 1.0

    class TwoBlockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block0Identity(), Block1Add()])
            self.cache_manager = None

    def build_mod(_m, x, _kw):
        return torch.zeros_like(x)

    def build_block0(_m, x, _kw):
        return _m.blocks[0](x)

    def per_post(_m, _i, x, _kw):
        return x

    device = torch.device("cpu")
    x0 = torch.zeros(2, 3, device=device)
    m = TwoBlockModel().to(device)
    cfg = CMConfig(num_steps=2, warmup=0, last_steps=0, enable_fb=True, fb_metric="residual_rel_l1")
    mgr = CacheManager(cfg); mgr.attach(num_steps=2); m.cache_manager = mgr

    # Step 1: compute with resume_from=1; block0 called once via builder
    mgr.begin_step("cond")
    out1 = run_with_cache_manager(model=m, x=x0, kwargs={}, build_mod_inp=build_mod, build_block0_residual=build_block0, per_block_post=per_post)
    assert torch.allclose(out1, x0 + 1)
    assert m.blocks[0].calls == 1

    # Step 2: skip should apply; block0 still called once to produce early signal
    mgr.begin_step("cond")
    out2 = run_with_cache_manager(model=m, x=x0, kwargs={}, build_mod_inp=build_mod, build_block0_residual=build_block0, per_block_post=per_post)
    assert torch.allclose(out2, x0)  # skip applied => unchanged
    assert m.blocks[0].calls == 2


def test_fb_downsample_affects_rel_hidden_metric():
    # Craft mod_inp that differs under downsample=1 vs 2
    device = torch.device("cpu")
    x = torch.zeros(1, 6, device=device)
    m = FakeModel().to(device)

    vals = [torch.ones_like(x), torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]], device=device)]
    step = {"i": 0}

    def mod_inp_seq(_m, _x, _kw):
        return vals[step["i"]]

    # Downsample=1
    cfg1 = CMConfig(num_steps=2, warmup=0, last_steps=0, enable_fb=True, fb_metric="hidden_rel_l1", fb_downsample=1)
    mgr1 = CacheManager(cfg1); mgr1.attach(num_steps=2); m.cache_manager = mgr1
    mgr1.begin_step("cond"); step["i"] = 0; _ = run_with_cache_manager(model=m, x=x, kwargs={}, build_mod_inp=mod_inp_seq)
    mgr1.begin_step("cond"); step["i"] = 1; d1 = mgr1.decide(x=x, mod_inp=mod_inp_seq(m, x, {}), x_after_block0=None)
    # mean goes from 1.0 to 0.5 => rel = 0.5
    assert pytest.approx(d1.rel, rel=1e-6) == 0.5

    # Downsample=2 (tokens 0,2,4 -> still 1.0)
    cfg2 = CMConfig(num_steps=2, warmup=0, last_steps=0, enable_fb=True, fb_metric="hidden_rel_l1", fb_downsample=2)
    mgr2 = CacheManager(cfg2); mgr2.attach(num_steps=2); m.cache_manager = mgr2
    mgr2.begin_step("cond"); step["i"] = 0; _ = run_with_cache_manager(model=m, x=x, kwargs={}, build_mod_inp=mod_inp_seq)
    mgr2.begin_step("cond"); step["i"] = 1; d2 = mgr2.decide(x=x, mod_inp=mod_inp_seq(m, x, {}), x_after_block0=None)
    # mean remains 1.0 => rel = 0.0
    assert pytest.approx(d2.rel, rel=1e-6) == 0.0


def test_decide_resume_from_block_for_residual_metric():
    # Verify decide() sets resume_from_block=1 for residual metric path
    device = torch.device("cpu")
    x = torch.zeros(1, 3, device=device)
    m = FakeModel().to(device)

    def build_block0(_m, x_in, _kw):
        return x_in  # identity => residual is zero

    cfg = CMConfig(num_steps=2, warmup=0, last_steps=0, enable_fb=True, fb_metric="residual_rel_l1")
    mgr = CacheManager(cfg); mgr.attach(num_steps=2); m.cache_manager = mgr
    # Step 1 compute (prev None)
    mgr.begin_step("cond"); _ = run_with_cache_manager(model=m, x=x, kwargs={}, build_mod_inp=build_mod_inp, build_block0_residual=build_block0)
    # Step 2 decide should set resume_from_block=1 and likely skip
    mgr.begin_step("cond")
    d = mgr.decide(x=x, mod_inp=None, x_after_block0=build_block0(m, x, {}))
    assert d.mode == "fb" and d.resume_from_block == 1


def test_pair_level_counters_mixed_sequence():
    # Mix forced compute, planned skip, and divergence
    device = torch.device("cpu")
    x = torch.zeros(1, 2, device=device)
    m = FakeModel().to(device)
    cfg = CMConfig(num_steps=2, warmup=1, last_steps=0, enable_tc=True)
    mgr = CacheManager(cfg); mgr.attach(num_steps=2); m.cache_manager = mgr

    # Step 1: forced compute due to warmup; run cond/uncond
    mgr.begin_step("cond"); _ = run_with_cache_manager(model=m, x=x, kwargs={}, build_mod_inp=build_mod_inp)
    mgr.begin_step("uncond"); _ = run_with_cache_manager(model=m, x=x, kwargs={}, build_mod_inp=build_mod_inp)

    # Step 2: cond skip planned, uncond diverges due to missing residual
    mgr.begin_step("cond"); _ = run_with_cache_manager(model=m, x=x, kwargs={}, build_mod_inp=build_mod_inp)
    mgr.uncond.residual = None; mgr.uncond.shape = None; mgr.uncond.dtype = None
    mgr.begin_step("uncond"); _ = run_with_cache_manager(model=m, x=x, kwargs={}, build_mod_inp=build_mod_inp)

    s = mgr.summary()
    assert s["pair"]["pair_total"] == 2.0
    assert s["pair"]["pair_forced_compute"] == 1.0
    assert s["pair"]["pair_planned_skips"] == 1.0
    assert s["pair"]["pair_divergence_failsafes"] >= 1.0


def test_priority_fb_over_tc_precedence():
    # When both FB and TC are enabled, FB takes precedence
    device = torch.device("cpu")
    x = torch.zeros(1, 4, device=device)
    m = FakeModel().to(device)
    cfg = CMConfig(num_steps=2, warmup=0, last_steps=0, enable_fb=True, enable_tc=True, fb_metric="hidden_rel_l1")
    mgr = CacheManager(cfg); mgr.attach(num_steps=2); m.cache_manager = mgr

    # Step 1: compute (prev None)
    mgr.begin_step("cond"); _ = run_with_cache_manager(model=m, x=x, kwargs={}, build_mod_inp=build_mod_inp)
    # Step 2: both would skip; ensure decision.mode == 'fb'
    mgr.begin_step("cond")
    d = mgr.decide(x=x, mod_inp=build_mod_inp(m, x, {}), x_after_block0=None)
    assert d.action == "skip" and d.mode == "fb"


def test_fb_ema_smoothing_hidden_metric():
    # Validate EMA smoothing on FB hidden metric
    device = torch.device("cpu")
    x = torch.zeros(1, 2, device=device)
    m = FakeModel().to(device)
    cfg = CMConfig(num_steps=3, warmup=0, last_steps=0, enable_fb=True, fb_metric="hidden_rel_l1", fb_ema=0.5, fb_thresh=1.0)
    mgr = CacheManager(cfg); mgr.attach(num_steps=3); m.cache_manager = mgr

    vals = [1.0, 1.1, 1.2]
    idx = {"i": 0}

    def mod_inp_seq(_m, x_in, _kw):
        v = vals[idx["i"]]
        return torch.ones_like(x_in) * v

    # Step 1: compute (prev None)
    mgr.begin_step("cond"); idx["i"] = 0
    _ = run_with_cache_manager(model=m, x=x, kwargs={}, build_mod_inp=mod_inp_seq)
    # Step 2: rel ≈ 0.1
    mgr.begin_step("cond"); idx["i"] = 1
    d2 = mgr.decide(x=x, mod_inp=mod_inp_seq(m, x, {}), x_after_block0=None)
    assert pytest.approx(d2.rel, rel=1e-4) == 0.1
    # Apply compute once to reset accum and set prev
    _ = run_with_cache_manager(model=m, x=x, kwargs={}, build_mod_inp=mod_inp_seq)
    # Step 3: rel unsmoothed ≈ 0.090909; smoothed ≈ 0.095454
    mgr.begin_step("cond"); idx["i"] = 2
    d3 = mgr.decide(x=x, mod_inp=mod_inp_seq(m, x, {}), x_after_block0=None)
    assert pytest.approx(d3.rel, rel=1e-4) == (0.5 * 0.1 + 0.5 * (0.1 / 1.1))


def test_cfg_uncond_reuse_reasons():
    device = torch.device("cpu")
    x = torch.zeros(1, 2, device=device)
    m = FakeModel().to(device)
    # Prepare a manager that will skip on step 2
    cfg = CMConfig(num_steps=2, warmup=0, last_steps=0, enable_tc=True)
    mgr = CacheManager(cfg); mgr.attach(num_steps=2); m.cache_manager = mgr

    # Step 1: compute cond/uncond
    mgr.begin_step("cond"); _ = run_with_cache_manager(model=m, x=x, kwargs={}, build_mod_inp=build_mod_inp)
    mgr.begin_step("uncond"); _ = run_with_cache_manager(model=m, x=x, kwargs={}, build_mod_inp=build_mod_inp)
    # Step 2 cond: decide skip
    mgr.begin_step("cond"); _ = run_with_cache_manager(model=m, x=x, kwargs={}, build_mod_inp=build_mod_inp)

    # cfg_sep_diff = False: uncond decision is reused with reason cfg_reuse
    mgr.cfg.cfg_sep_diff = False
    mgr.begin_step("uncond")
    d_u = mgr.decide(x=x, mod_inp=build_mod_inp(m, x, {}), x_after_block0=None)
    assert d_u.reason == "cfg_reuse"

    # cfg_sep_diff = True: uncond computes metrics but follows cond action; reason cfg_follow_cond
    mgr.cfg.cfg_sep_diff = True
    mgr.begin_step("uncond")
    d_u2 = mgr.decide(x=x, mod_inp=build_mod_inp(m, x, {}), x_after_block0=None)
    assert d_u2.reason == "cfg_follow_cond"


def test_summary_counts_and_rates():
    device = torch.device("cpu")
    x = torch.zeros(1, 2, device=device)
    m = FakeModel().to(device)
    cfg = CMConfig(num_steps=2, warmup=0, last_steps=0, enable_tc=True)
    mgr = CacheManager(cfg); mgr.attach(num_steps=2); m.cache_manager = mgr

    # Step 1: compute, Step 2: skip
    mgr.begin_step("cond"); _ = run_with_cache_manager(model=m, x=x, kwargs={}, build_mod_inp=build_mod_inp)
    mgr.begin_step("cond"); _ = run_with_cache_manager(model=m, x=x, kwargs={}, build_mod_inp=build_mod_inp)
    s = mgr.summary()
    assert s["cond"]["total"] == 2.0
    assert s["cond"]["skipped"] == 1.0
    assert pytest.approx(s["cond"]["skip_rate"], rel=1e-6) == 50.0


def test_runner_hidden_metric_skip_no_block0_rerun():
    # With hidden metric gating, a skip should not rerun block0
    class Block0Counting(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            self.calls += 1
            return x

    class Block1Add(torch.nn.Module):
        def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            return x + 1

    class Model2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block0Counting(), Block1Add()])
            self.cache_manager = None

    device = torch.device("cpu")
    x = torch.zeros(1, 2, device=device)
    m = Model2().to(device)
    cfg = CMConfig(num_steps=2, warmup=0, last_steps=0, enable_fb=True, fb_metric="hidden_rel_l1")
    mgr = CacheManager(cfg); mgr.attach(num_steps=2); m.cache_manager = mgr

    # Step 1: compute → block0 called once
    mgr.begin_step("cond")
    out1 = run_with_cache_manager(model=m, x=x, kwargs={}, build_mod_inp=build_mod_inp, build_block0_residual=None)
    assert torch.allclose(out1, x + 1)
    assert m.blocks[0].calls == 1

    # Step 2: skip applies; block0 is not re-run
    mgr.begin_step("cond")
    out2 = run_with_cache_manager(model=m, x=x, kwargs={}, build_mod_inp=build_mod_inp, build_block0_residual=None)
    assert torch.allclose(out2, x)
    assert m.blocks[0].calls == 1
