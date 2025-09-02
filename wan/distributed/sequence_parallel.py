# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.cuda.amp as amp

from ..modules.model import sinusoidal_embedding_1d
from .ulysses import distributed_attention
from .util import gather_forward, get_rank, get_world_size
from ..utils.teacache import summarize_mod, rescale


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@torch.amp.autocast('cuda', enabled=False)
# OPTIM: Apply RoPE per-rank with pre-padding to align shards; fp64 complex multiply then cast back.
def rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        sp_size = get_world_size()
        sp_rank = get_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


def sp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == 'i2v':
        assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast('cuda', dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t).unflatten(0, (bt, seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)

    # TeaCache gate for SP: compute first-block modulated input and decide skip/compute consistently across ranks.
    use_fbcache = bool(getattr(self, "enable_fbcache", False)) and getattr(
        self, "fbcache", None) is not None
    use_teacache = (not use_fbcache) and bool(getattr(self, "enable_teacache", False)) and getattr(
        self, "teacache", None) is not None

    if not (use_teacache or use_fbcache):
        # Baseline compute path (unchanged)
        for block in self.blocks:
            x = block(x, **kwargs)
    elif use_teacache:
        block0 = self.blocks[0]
        norm_x = block0.norm1(x).float()
        with amp.autocast(dtype=torch.float32):
            e_chunks = (block0.modulation.unsqueeze(0) + kwargs["e"]).chunk(
                6, dim=2)
        mod_inp = norm_x * (1 + e_chunks[1].squeeze(2)) + e_chunks[0].squeeze(2)

        state = getattr(self, "teacache")  # type: ignore[attr-defined]
        branch_state = state.branch_state()

        # Lifecycle guards
        force_compute = False
        if branch_state.prev_mod_sig is None:
            force_compute = True
        if state.branch == 'cond':
            if state.cnt < state.warmup:
                force_compute = True
            if state.cnt >= max(0, state.num_steps - state.last_steps):
                force_compute = True
            # Optional alternating guard
            if bool(getattr(self, "alternating_teacache", False)):
                if (state.cnt % 2) == 1:
                    force_compute = True

        skip = False
        cur_sig = summarize_mod(mod_inp)
        if not force_compute and branch_state.prev_mod_sig is not None:
            prev = branch_state.prev_mod_sig
            import math
            rel = abs(cur_sig - prev) / (abs(prev) + 1e-8)
            # AllReduce mean across SP ranks for a unified decision
            rel_t = torch.tensor([rel], device=norm_x.device, dtype=torch.float32)
            import torch.distributed as dist
            dist.all_reduce(rel_t, op=dist.ReduceOp.SUM)
            world = get_world_size()
            rel = float(rel_t.item() / max(1, int(world)))
            rel_rescaled = rescale(rel, state.policy)
            if not (math.isfinite(rel) and math.isfinite(rel_rescaled)):
                state.failsafe_count += 1
                force_compute = True
            if not force_compute:
                branch_state.accum += float(rel_rescaled)
                # Telemetry
                branch_state.sum_rel += float(rel)
                branch_state.sum_rescaled += float(rel_rescaled)
                branch_state.count_rel += 1
                skip = branch_state.accum < float(state.thresh)

        if skip and (branch_state.prev_residual is not None):
            # Residual is already sequence-sharded; ensure shape match
            if (branch_state.shape is not None and x.shape == tuple(
                    branch_state.shape)):
                x = x + branch_state.prev_residual.to(x.device, x.dtype)
                branch_state.skipped += 1
            else:
                x_before = x
                for block in self.blocks:
                    x = block(x, **kwargs)
                branch_state.prev_residual = (x - x_before).detach().to(x.dtype)
                branch_state.shape = tuple(branch_state.prev_residual.shape)
                branch_state.accum = 0.0
        else:
            x_before = x
            for block in self.blocks:
                x = block(x, **kwargs)
            branch_state.prev_residual = (x - x_before).detach().to(x.dtype)
            branch_state.shape = tuple(branch_state.prev_residual.shape)
            branch_state.accum = 0.0

        branch_state.prev_mod_sig = float(cur_sig)
        branch_state.total += 1
    elif use_fbcache:
        # ------------------------ FBCache gating (SP path) ------------------------
        # Compute early signal and reduce a scalar across SP ranks for a unified decision.
        from ..utils.fbcache import (
            summarize_hidden,
            summarize_residual,
            rescale_metric,
        )
        fb_state = getattr(self, "fbcache")  # type: ignore[attr-defined]
        fb_branch = fb_state.branch_state()
        block0 = self.blocks[0]
        norm_x = block0.norm1(x).float()
        with amp.autocast(dtype=torch.float32):
            e_chunks = (block0.modulation.unsqueeze(0) + kwargs["e"]).chunk(6, dim=2)
        mod_inp = norm_x * (1 + e_chunks[1].squeeze(2)) + e_chunks[0].squeeze(2)

        force_compute = False
        if fb_branch.prev_sig is None:
            force_compute = True
        if fb_state.branch == 'cond':
            if fb_state.cnt < fb_state.warmup:
                force_compute = True
            if fb_state.cnt >= max(0, fb_state.num_steps - fb_state.last_steps):
                force_compute = True

        # Compute current signature (metric dependent). For residual metric, compute block0 to get residual.
        cur_sig = None
        x_after_block0 = None
        if fb_state.metric == 'hidden_rel_l1' or fb_state.metric == 'hidden_rel_l2':
            cur_sig = summarize_hidden(mod_inp, fb_state.downsample)
        elif fb_state.metric == 'residual_rel_l1':
            x0_before = x
            x_after_block0 = block0(x0_before, **kwargs)
            r1 = (x_after_block0 - x0_before)
            cur_sig = summarize_hidden(r1, fb_state.downsample)
        else:
            cur_sig = summarize_hidden(mod_inp, fb_state.downsample)

        import math
        rel = 0.0
        rel_rescaled = 0.0
        if not force_compute and fb_branch.prev_sig is not None:
            if not fb_state.cfg_sep_diff and fb_state.branch == 'uncond' and fb_state.last_cond_rel is not None:
                rel = fb_state.last_cond_rel
                rel_rescaled = fb_state.last_cond_rescaled if fb_state.last_cond_rescaled is not None else rel
            else:
                prev = fb_branch.prev_sig
                rel = abs(cur_sig - prev) / (abs(prev) + 1e-8)
                if fb_state.ema > 0.0:
                    if fb_branch.ema_val is None:
                        fb_branch.ema_val = rel
                    else:
                        fb_branch.ema_val = fb_state.ema * fb_branch.ema_val + (1.0 - fb_state.ema) * rel
                    rel = fb_branch.ema_val
                rel_rescaled = rescale_metric(rel, 'linear')
            if not (math.isfinite(rel)):
                fb_state.failsafe_count += 1
                force_compute = True

        # All-reduce scalar across SP ranks for unified decision
        rel_t = torch.tensor([rel_rescaled], device=norm_x.device, dtype=torch.float32)
        dist.all_reduce(rel_t, op=dist.ReduceOp.SUM)
        world = get_world_size()
        rel_rescaled = float(rel_t.item() / max(1, int(world)))

        skip = False
        if not force_compute and fb_branch.prev_sig is not None:
            fb_branch.accum += float(rel_rescaled)
            fb_branch.sum_rel += float(rel)
            fb_branch.sum_rescaled += float(rel_rescaled)
            fb_branch.count_rel += 1
            skip = fb_branch.accum < float(fb_state.thresh)

        if skip and (fb_branch.prev_residual is not None):
            if (fb_branch.shape is not None and x.shape == tuple(fb_branch.shape)):
                x = x + fb_branch.prev_residual.to(x.device, x.dtype)
                fb_branch.skipped += 1
            else:
                x_before = x
                if x_after_block0 is None:
                    for block in self.blocks:
                        x = block(x, **kwargs)
                else:
                    x = x_after_block0
                    for block in self.blocks[1:]:
                        x = block(x, **kwargs)
                fb_branch.prev_residual = (x - x_before).detach().to(x.dtype)
                fb_branch.shape = tuple(fb_branch.prev_residual.shape)
                fb_branch.accum = 0.0
        else:
            x_before = x
            if x_after_block0 is None:
                for block in self.blocks:
                    x = block(x, **kwargs)
            else:
                x = x_after_block0
                for block in self.blocks[1:]:
                    x = block(x, **kwargs)
            fb_branch.prev_residual = (x - x_before).detach().to(x.dtype)
            fb_branch.shape = tuple(fb_branch.prev_residual.shape)
            fb_branch.accum = 0.0

        fb_branch.prev_sig = float(cur_sig)
        if fb_state.branch == 'cond':
            fb_state.last_cond_rel = float(rel)
            fb_state.last_cond_rescaled = float(rel_rescaled)
        fb_branch.total += 1

    # head
    x = self.head(x, e)

    # Context Parallel
    x = gather_forward(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def sp_attn_forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)

    x = distributed_attention(
        half(q),
        half(k),
        half(v),
        seq_lens,
        window_size=self.window_size,
    )

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x
