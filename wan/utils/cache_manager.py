"""Cache Manager for TeaCache, FBCache, and CFG-aware control (inference-only).

Implements a unified lifecycle and decision policy with minimal configuration:
- Lifecycle: init -> warmup -> main -> last-steps -> reset
- Branches: cond/uncond (separate-CFG for Wan2.2; fused-CFG optional future)
- Priority: FBCache -> TeaCache -> Compute
- Distributed: SP mean reduction of scalar metrics with guarded fail-safes
- Offload: explicit CPU<->GPU moves for cached residuals
- Telemetry: per-branch counters + pair-level CFG stats + global failsafes

This module contains no training-time logic. All math is inference-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

import math
import torch

Branch = Literal["cond", "uncond"]
Mode = Literal["fb", "tc"]


@dataclass
class CMConfig:
    """Minimal, immutable config for a run (set once at attach())."""

    # Lifecycle
    num_steps: int
    warmup: int = 1
    last_steps: int = 1

    # TeaCache
    enable_tc: bool = False
    tc_thresh: float = 0.08
    tc_policy: str = "linear"

    # FBCache
    enable_fb: bool = False
    fb_thresh: float = 0.08
    fb_metric: str = "hidden_rel_l1"  # or 'residual_rel_l1', 'hidden_rel_l2'
    fb_downsample: int = 1
    fb_ema: float = 0.0

    # CFG cache (pair-level policy)
    cfg_sep_diff: bool = False  # default: cond metrics/action reused for uncond

    # Priority & distributed
    evaluation_order: Tuple[Mode, ...] = ("fb", "tc")
    sp_world_size: int = 1
    # Optional explicit process group for SP reductions
    sp_group: Optional[Any] = None


@dataclass
class Decision:
    """Final decision for a branch at a step."""

    action: Literal["skip", "compute"]
    mode: Optional[Mode]
    resume_from_block: int = 0
    reason: str = ""
    rel: float = 0.0
    rel_rescaled: float = 0.0


@dataclass
class _BranchState:
    """Per-branch state shared across modes."""

    residual: Optional[torch.Tensor] = None
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[torch.dtype] = None

    tc_prev_sig: Optional[float] = None
    tc_accum: float = 0.0

    fb_prev_sig: Optional[float] = None
    fb_accum: float = 0.0
    fb_ema_val: Optional[float] = None

    total: int = 0
    skipped: int = 0
    sum_rel: float = 0.0
    sum_rescaled: float = 0.0
    count_rel: int = 0


class CacheManager:
    """Unified cache manager coordinating TeaCache/FBCache gating and CFG cache."""

    def __init__(self, config: CMConfig):
        self.cfg = config
        self.branch: Branch = "cond"
        self.cnt: int = 0  # executed step counter (cond only)
        self.cond = _BranchState()
        self.uncond = _BranchState()
        self.failsafe_count: int = 0
        # Last cond metrics for pair reuse
        self._last_cond_rel: Dict[Mode, Optional[float]] = {"fb": None, "tc": None}
        self._last_cond_rescaled: Dict[Mode, Optional[float]] = {"fb": None, "tc": None}
        self._last_cond_decision: Optional[Decision] = None
        # Pair-level stats
        self.pair_total: int = 0
        self.pair_planned_skips: int = 0
        self.pair_forced_compute: int = 0
        self.pair_divergence_failsafes: int = 0

    # ---------- Lifecycle ----------
    def attach(self, num_steps: Optional[int] = None, sp_world_size: Optional[int] = None) -> None:
        if num_steps is not None:
            self.cfg.num_steps = int(num_steps)
        if sp_world_size is not None:
            self.cfg.sp_world_size = int(sp_world_size)
        self.reset()

    def reset(self) -> None:
        self.cnt = 0
        self.cond = _BranchState()
        self.uncond = _BranchState()
        self.failsafe_count = 0
        self._last_cond_rel = {"fb": None, "tc": None}
        self._last_cond_rescaled = {"fb": None, "tc": None}
        self._last_cond_decision = None
        self.pair_total = 0
        self.pair_planned_skips = 0
        self.pair_forced_compute = 0
        self.pair_divergence_failsafes = 0

    def begin_step(self, branch: Branch) -> None:
        # Track pair lifecycle at cond entry
        self.branch = branch
        if branch == "cond":
            self.cnt += 1
            self.pair_total += 1

    # ---------- Offload ----------
    def move_cached_residuals_to(self, device: torch.device) -> None:
        for br in (self.cond, self.uncond):
            if br.residual is not None and br.residual.device != device:
                try:
                    br.residual = br.residual.to(device)
                except RuntimeError:
                    # OOM or device move failure: drop residual for safety
                    self.failsafe_count += 1
                    br.residual = None
                    br.shape = None
                    br.dtype = None

    # ---------- Internals ----------
    def _br(self) -> _BranchState:
        return self.cond if self.branch == "cond" else self.uncond

    @staticmethod
    def _mean_abs(x: torch.Tensor, downsample: int = 1) -> float:
        if downsample > 1:
            x = x[:, ::downsample]
        return float(x.abs().mean().detach().to("cpu"))

    @staticmethod
    def _rescale(v: float, policy: str) -> float:
        # Only linear supported; conservative fallback for unknowns.
        return v

    def _sp_mean(self, value: float, device: torch.device) -> float:
        """Average a scalar across the active distributed group if initialized.

        Uses the actual group world size to normalize instead of relying on the
        configured `sp_world_size`, which may be stale or mismatched.
        """
        try:
            import torch.distributed as dist
            if not dist.is_initialized():
                return value
            t = torch.tensor([value], dtype=torch.float32, device=device)
            try:
                # Use explicit SP group if provided
                group = getattr(self.cfg, "sp_group", None)
                dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
                ws = dist.get_world_size(group=group) if group is not None else dist.get_world_size()
            except Exception:
                # Fallback to default group if group ops fail
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                ws = dist.get_world_size()
            ws = max(1, int(ws))
            return float(t.item() / ws)
        except Exception:
            # Any reduction failure forces local compute downstream via failsafe
            self.failsafe_count += 1
            return value

    # ---------- Decision ----------
    def decide(self, x: torch.Tensor, mod_inp: Optional[torch.Tensor], x_after_block0: Optional[torch.Tensor] = None) -> Decision:
        """Compute a decision for the active branch.

        CFG semantics (separate-CFG):
        - On 'cond': compute decision normally per priority, store cond action/metrics for reuse.
        - On 'uncond': if cfg_sep_diff is False (default), reuse cond rescaled metrics and action; if True, compute metrics but still follow cond action unless a hard failsafe occurs at apply().
        """
        br = self._br()

        # Determine lifecycle phase for cond (warmup/last-steps).
        force_compute = False
        if self.branch == "cond":
            # Warmup: force compute only when warmup>0 and within first K executed steps
            if self.cfg.warmup > 0 and self.cnt <= self.cfg.warmup:
                force_compute = True
            # Last-steps: force compute only when last_steps>0 and within final K steps
            if self.cfg.last_steps > 0 and self.cnt > max(0, self.cfg.num_steps - self.cfg.last_steps):
                force_compute = True

        # Uncond path under CFG cache: reuse cond decision/metric, but guard skip readiness
        if self.branch == "uncond" and not self.cfg.cfg_sep_diff and self._last_cond_decision is not None:
            d = self._last_cond_decision
            # Count per-branch total; skip increments are accounted in apply()
            br.total += 1
            # If cond decided to skip, ensure this branch has a valid residual before honoring it.
            if d.action == "skip":
                ready = (
                    br.residual is not None
                    and br.shape is not None
                    and tuple(br.shape) == tuple(x.shape)
                    and (br.dtype is None or br.dtype == x.dtype)
                )
                if not ready:
                    # Divergence: uncond cannot follow cond skip; force compute and track.
                    self.pair_divergence_failsafes += 1
                    return Decision(
                        action="compute",
                        mode=d.mode,
                        resume_from_block=d.resume_from_block,
                        reason="cfg_residual_guard",
                        rel=d.rel,
                        rel_rescaled=d.rel_rescaled,
                    )
            return Decision(
                action=d.action,
                mode=d.mode,
                resume_from_block=d.resume_from_block,
                reason="cfg_reuse",
                rel=d.rel,
                rel_rescaled=d.rel_rescaled,
            )

        last_decision: Optional[Decision] = None

        for mode in self.cfg.evaluation_order:
            if mode == "fb" and not self.cfg.enable_fb:
                continue
            if mode == "tc" and not self.cfg.enable_tc:
                continue

            if mode == "tc":
                if mod_inp is None:
                    continue
                cur_sig = self._mean_abs(mod_inp)
                prev = br.tc_prev_sig
                rel = 0.0
                if prev is not None:
                    rel = abs(cur_sig - prev) / (abs(prev) + 1e-8)
                    rel = self._sp_mean(rel, device=mod_inp.device)
                rel_rescaled = self._rescale(rel, self.cfg.tc_policy)

                # Telemetry
                if prev is not None:
                    br.sum_rel += float(rel)
                    br.sum_rescaled += float(rel_rescaled)
                    br.count_rel += 1

                # Accumulate and decide: add rescaled value, then compare to threshold
                if prev is not None and not force_compute:
                    br.tc_accum += float(rel_rescaled)
                skip = (not force_compute) and (prev is not None) and (br.tc_accum < float(self.cfg.tc_thresh))

                # Guard: ensure residual exists and matches shape/dtype before allowing skip
                if skip:
                    ready = (br.residual is not None and br.shape is not None and tuple(br.shape) == tuple(x.shape)
                             and (br.dtype is None or br.dtype == x.dtype))
                    if not ready:
                        skip = False
                        rel_reason = "residual_guard"
                    else:
                        rel_reason = "acc<tc_thresh"
                else:
                    rel_reason = "forced" if force_compute else "tc>=thresh"

                # Update sig for next step
                br.tc_prev_sig = float(cur_sig)

                d = Decision(action=("skip" if skip else "compute"), mode="tc", resume_from_block=0, reason=rel_reason, rel=float(rel), rel_rescaled=float(rel_rescaled))
                last_decision = d
                if skip:
                    break

            elif mode == "fb":
                metric = self.cfg.fb_metric
                if metric.startswith("hidden"):
                    if mod_inp is None:
                        continue
                    cur_sig = self._mean_abs(mod_inp, downsample=max(1, int(self.cfg.fb_downsample)))
                    prev = br.fb_prev_sig
                    rel = 0.0
                    if prev is not None:
                        if metric.endswith("_l2"):
                            rel = (cur_sig - prev) ** 2 / (abs(prev) + 1e-8)
                        else:
                            rel = abs(cur_sig - prev) / (abs(prev) + 1e-8)
                        rel = self._sp_mean(rel, device=mod_inp.device)
                elif metric.startswith("residual"):
                    if x_after_block0 is None:
                        continue
                    residual = x_after_block0 - x
                    cur_sig = self._mean_abs(residual, downsample=max(1, int(self.cfg.fb_downsample)))
                    prev = br.fb_prev_sig
                    rel = 0.0
                    if prev is not None:
                        rel = abs(cur_sig - prev) / (abs(prev) + 1e-8)
                        rel = self._sp_mean(rel, device=residual.device)
                else:
                    continue

                # EMA smoothing (optional)
                if self.cfg.fb_ema > 0.0:
                    if br.fb_ema_val is None:
                        br.fb_ema_val = rel
                    else:
                        br.fb_ema_val = self.cfg.fb_ema * br.fb_ema_val + (1.0 - self.cfg.fb_ema) * rel
                    rel = float(br.fb_ema_val)

                rel_rescaled = self._rescale(rel, policy="linear")

                # Telemetry
                if prev is not None:
                    br.sum_rel += float(rel)
                    br.sum_rescaled += float(rel_rescaled)
                    br.count_rel += 1

                if prev is not None and not force_compute:
                    br.fb_accum += float(rel_rescaled)
                skip = (not force_compute) and (prev is not None) and (br.fb_accum < float(self.cfg.fb_thresh))

                if skip:
                    ready = (br.residual is not None and br.shape is not None and tuple(br.shape) == tuple(x.shape)
                             and (br.dtype is None or br.dtype == x.dtype))
                    if not ready:
                        skip = False
                        fb_reason = "residual_guard"
                    else:
                        fb_reason = "acc<fb_thresh"
                else:
                    fb_reason = "forced" if force_compute else "fb>=thresh"

                br.fb_prev_sig = float(cur_sig)
                resume_from = 1 if metric.startswith("residual") and x_after_block0 is not None else 0
                d = Decision(action=("skip" if skip else "compute"), mode="fb", resume_from_block=resume_from, reason=fb_reason, rel=float(rel), rel_rescaled=float(rel_rescaled))
                last_decision = d
                if skip:
                    break

        # Bookkeep branch totals
        br.total += 1

        # Record pair-level info on cond
        if self.branch == "cond" and last_decision is not None:
            self._last_cond_decision = last_decision
            # Store for CFG reuse
            if last_decision.mode in ("fb", "tc"):
                self._last_cond_rel[last_decision.mode] = float(last_decision.rel)
                self._last_cond_rescaled[last_decision.mode] = float(last_decision.rel_rescaled)
            if last_decision.action == "skip":
                self.pair_planned_skips += 1
            if last_decision.reason == "forced":
                self.pair_forced_compute += 1

        # If nothing decided yet, synthesize a compute decision
        d = last_decision or Decision(action="compute", mode=None, resume_from_block=0, reason="no-mode")

        # Separate-CFG with cfg_sep_diff=true: compute metrics for uncond but still follow cond action
        if self.branch == "uncond" and self.cfg.cfg_sep_diff and self._last_cond_decision is not None:
            cond_d = self._last_cond_decision
            d = Decision(
                action=cond_d.action,
                mode=cond_d.mode,
                resume_from_block=cond_d.resume_from_block,
                reason="cfg_follow_cond",
                rel=d.rel,
                rel_rescaled=d.rel_rescaled,
            )
        return d

    # ---------- Application & Update ----------
    def apply(self, decision: Decision, x: torch.Tensor) -> Tuple[torch.Tensor, int, bool]:
        br = self._br()
        if decision.action != "skip":
            return x, decision.resume_from_block, False

        # Guard: residual existence, shape, and dtype match
        if (
            br.residual is None
            or br.shape is None
            or tuple(br.shape) != tuple(x.shape)
            or (br.dtype is not None and br.dtype != x.dtype)
        ):
            # Pair-consistency failsafe if uncond cannot follow cond skip
            if self.branch == "uncond":
                self.pair_divergence_failsafes += 1
            self.failsafe_count += 1
            # Signal to caller that skip was not applied so it can fall back to compute
            return x, decision.resume_from_block, False

        res = br.residual.to(device=x.device, dtype=x.dtype)
        x_out = x + res
        br.skipped += 1
        return x_out, decision.resume_from_block, True

    def update(self, decision: Decision, x_before: torch.Tensor, x_after: torch.Tensor) -> None:
        br = self._br()
        res = (x_after - x_before).detach().to(x_after.dtype)
        br.residual = res
        br.shape = tuple(res.shape)
        br.dtype = res.dtype
        # Reset the deciding mode accumulator
        if decision.mode == "tc":
            br.tc_accum = 0.0
        elif decision.mode == "fb":
            br.fb_accum = 0.0
        else:
            br.tc_accum = 0.0
            br.fb_accum = 0.0

    # ---------- Telemetry ----------
    def summary(self) -> Dict[str, Dict[str, float]]:
        def pack(br: _BranchState) -> Dict[str, float]:
            avg_rel = (br.sum_rel / br.count_rel) if br.count_rel else 0.0
            avg_rescaled = (br.sum_rescaled / br.count_rel) if br.count_rel else 0.0
            return dict(
                total=float(br.total),
                skipped=float(br.skipped),
                skip_rate=(100.0 * br.skipped / br.total) if br.total else 0.0,
                avg_rel=float(avg_rel),
                avg_rescaled=float(avg_rescaled),
            )

        return {
            "cond": pack(self.cond),
            "uncond": pack(self.uncond),
            "pair": {
                "pair_total": float(self.pair_total),
                "pair_planned_skips": float(self.pair_planned_skips),
                "pair_forced_compute": float(self.pair_forced_compute),
                "pair_divergence_failsafes": float(self.pair_divergence_failsafes),
            },
            "failsafe_count": {"value": float(self.failsafe_count)},
            "config": {
                "num_steps": float(self.cfg.num_steps),
                "warmup": float(self.cfg.warmup),
                "last_steps": float(self.cfg.last_steps),
                "tc_enabled": float(self.cfg.enable_tc),
                "fb_enabled": float(self.cfg.enable_fb),
                "cfg_sep_diff": float(self.cfg.cfg_sep_diff),
                "priority_fb_first": float(int(self.cfg.evaluation_order[0] == "fb")),
                "sp_world_size": float(self.cfg.sp_world_size),
            },
        }
