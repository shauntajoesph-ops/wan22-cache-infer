# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
# OPTIM: Keep RoPE computation in fp32 to avoid numerical drift; compute once and reuse across blocks.
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
# OPTIM: Apply RoPE in fp64 for complex multiply accuracy, then cast back; pre-slices per-dimension to minimize work.
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # OPTIM: Precompute multipliers only for needed seq_len and broadcast over H/W with view to avoid expands in GEMM.
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(
                self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        # initialize weights
        self.init_weights()

        # TeaCache integration (Phase 1):
        # - Disabled by default; pipelines attach a `TeaCacheState` and flip this flag.
        # - We avoid importing here to keep model portable without the feature.
        self.enable_teacache = False  # type: ignore[attr-defined]
        self.teacache = None  # type: ignore[attr-defined]

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
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
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
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
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        # TeaCache gate: conditionally skip the block stack by reusing a cached residual.
        # The gating signal is derived from block0's norm-and-modulate input.
        # FB Cache: alternative gating using first-block oriented signal.
        # Both gates are mutually exclusive by policy; if FBCache is enabled, it takes precedence.
        # Default behavior remains identical to baseline when both gates are disabled.
        use_fbcache = bool(getattr(self, "enable_fbcache", False)) and getattr(
            self, "fbcache", None) is not None
        # This block is guarded and has no effect unless the corresponding state is set.
        use_teacache = (not use_fbcache) and bool(getattr(self, "enable_teacache", False)) and getattr(
            self, "teacache", None) is not None

        if not (use_teacache or use_fbcache):
            # Baseline path (unchanged): compute all blocks.
            for block in self.blocks:
                x = block(x, **kwargs)
        elif use_teacache:
            # Compute first-block modulated input as in WanAttentionBlock.forward
            block0 = self.blocks[0]
            # norm1(x) in fp32 for numerical stability
            norm_x = block0.norm1(x).float()  # [B, L, C]
            # time modulation chunks: (bias, scale, gate) groupings taken from design (6 parts)
            with torch.amp.autocast('cuda', dtype=torch.float32):
                e_chunks = (block0.modulation.unsqueeze(0) + kwargs["e"]).chunk(
                    6, dim=2)
            # modulated input used by the first self-attention (pre-attn resid)
            mod_inp = norm_x * (1 + e_chunks[1].squeeze(2)) + e_chunks[0].squeeze(2)

            # Import locally to avoid circular deps at import time
            from ..utils.teacache import summarize_mod, rescale, move_residual_to

            state = getattr(self, "teacache")  # type: ignore[attr-defined]
            assert state is not None
            # Select per-branch state: cond/uncond
            branch_state = state.branch_state()

            # Determine if we should force compute due to lifecycle rules
            force_compute = False
            # First use: no previous signature
            if branch_state.prev_mod_sig is None:
                force_compute = True
            # Warmup / last steps: always compute
            if state.branch == 'cond':
                # Only cond branch increments cnt externally; here we read it.
                if state.cnt < state.warmup:
                    force_compute = True
                if state.cnt >= max(0, state.num_steps - state.last_steps):
                    force_compute = True
                # Optional alternating guard: only every other executed step eligible to skip
                if bool(getattr(self, "alternating_teacache", False)):
                    if (state.cnt % 2) == 1:
                        force_compute = True

            # Compute relative change if possible
            skip = False
            cur_sig = summarize_mod(mod_inp)
            if not force_compute and branch_state.prev_mod_sig is not None:
                prev = branch_state.prev_mod_sig
                # Relative L1 (scalar) with epsilon to avoid div-by-zero
                import math
                rel = abs(cur_sig - prev) / (abs(prev) + 1e-8)
                # Sequence parallel synchronization: if using SP, all ranks must agree on the scalar
                if hasattr(self, "sp_size") and getattr(self, "sp_size") > 1:
                    import torch.distributed as dist
                    # Convert to tensor on the correct device for reduction
                    rel_t = torch.tensor([rel], device=norm_x.device, dtype=torch.float32)
                    # AllReduce SUM, then divide by world size for mean
                    dist.all_reduce(rel_t, op=dist.ReduceOp.SUM)
                    world = getattr(self, "sp_size")
                    rel = float(rel_t.item() / max(1, int(world)))
                # Rescale according to policy (identity for 'linear')
                rel_rescaled = rescale(rel, state.policy)
                # Fail-safe: invalid numbers cause a forced compute and reset.
                if not (math.isfinite(rel) and math.isfinite(rel_rescaled)):
                    state.failsafe_count += 1  # type: ignore[attr-defined]
                    force_compute = True
                # Accumulate and decide
                if not force_compute:
                    branch_state.accum += float(rel_rescaled)
                    # Telemetry: track observed rels
                    branch_state.sum_rel += float(rel)
                    branch_state.sum_rescaled += float(rel_rescaled)
                    branch_state.count_rel += 1
                    skip = branch_state.accum < float(state.thresh)

            if skip and (branch_state.prev_residual is not None):
                # Skip compute: add cached residual to current hidden states.
                # Ensure residual is on the right device and dtype.
                target_dtype = x.dtype
                target_device = norm_x.device
                res = move_residual_to(branch_state.prev_residual, target_device,
                                       target_dtype)
                # Shape/dtype guards: if mismatch, fall back to compute path
                if (branch_state.shape is not None and
                        tuple(res.shape) == tuple(branch_state.shape)):
                    x = x + res
                    branch_state.skipped += 1  # record skip
                else:
                    # Fallback: compute blocks and refresh residual
                    state.failsafe_count += 1  # type: ignore[attr-defined]
                    x_before = x
                    for block in self.blocks:
                        x = block(x, **kwargs)
                    # Cache residual tensor in model compute dtype
                    branch_state.prev_residual = (x - x_before).detach().to(
                        target_dtype)
                    branch_state.shape = tuple(branch_state.prev_residual.shape)
                    branch_state.accum = 0.0  # reset on compute
            else:
                # Compute path (forced or no valid residual)
                x_before = x
                for block in self.blocks:
                    x = block(x, **kwargs)
                # Cache residual tensor in model compute dtype
                branch_state.prev_residual = (x - x_before).detach().to(x.dtype)
                branch_state.shape = tuple(branch_state.prev_residual.shape)
                branch_state.accum = 0.0  # reset on compute

            # Update signature for next step (stored as CPU float)
            branch_state.prev_mod_sig = float(cur_sig)
            # Telemetry: total gating decisions for this branch
            branch_state.total += 1

        elif use_fbcache:
            # ------------------------ FBCache gating path ------------------------
            # We compute an early, low-cost signal tied to the first block to decide
            # whether to reuse the previous step's full-stack residual.
            fb_state = getattr(self, "fbcache")  # type: ignore[attr-defined]
            assert fb_state is not None
            fb_branch = fb_state.branch_state()

            # Pre-compute first-block tensors used for hidden-based metric.
            block0 = self.blocks[0]
            norm_x = block0.norm1(x).float()  # Stable normalization in fp32
            with torch.amp.autocast('cuda', dtype=torch.float32):
                e_chunks = (block0.modulation.unsqueeze(0) + kwargs["e"]).chunk(
                    6, dim=2)
            mod_inp = norm_x * (1 + e_chunks[1].squeeze(2)) + e_chunks[0].squeeze(2)

            # Import helpers locally to avoid circular imports at module import time.
            from ..utils.fbcache import (
                summarize_hidden,
                summarize_residual,
                rescale_metric,
                move_residual_to as fb_move_residual_to,
            )

            # Lifecycle guards: force compute when state is uninitialized, within warmup,
            # or within the last K executed steps. Only the cond branch increments cnt.
            force_compute = False
            if fb_branch.prev_sig is None:
                force_compute = True
            if fb_state.branch == 'cond':
                if fb_state.cnt < fb_state.warmup:
                    force_compute = True
                if fb_state.cnt >= max(0, fb_state.num_steps - fb_state.last_steps):
                    force_compute = True

            # Compute current step's scalar signature according to selected metric.
            # For 'residual_rel_l1', we need the first-block output to build the residual.
            cur_sig = None
            x_after_block0 = None  # used if we later decide to compute and want to reuse work
            if fb_state.metric == 'hidden_rel_l1' or fb_state.metric == 'hidden_rel_l2':
                cur_sig = summarize_hidden(mod_inp, fb_state.downsample)
            elif fb_state.metric == 'residual_rel_l1':
                # Run block0 to obtain its output; compute residual r1 = out0 - x.
                x0_before = x
                x_after_block0 = block0(x0_before, **kwargs)
                r1 = (x_after_block0 - x0_before)
                cur_sig = summarize_residual(r1, fb_state.downsample)
            else:
                # Fallback to hidden-based metric for unknown identifiers (conservative)
                cur_sig = summarize_hidden(mod_inp, fb_state.downsample)

            # Derive relative change vs previous signature. Optionally reuse cond diff for CFG
            # when cfg_sep_diff is False to keep within-step consistency.
            import math
            if not force_compute and fb_branch.prev_sig is not None:
                if not fb_state.cfg_sep_diff and fb_state.branch == 'uncond' and fb_state.last_cond_rel is not None:
                    # Reuse previously computed cond diff and rescaled values.
                    rel = fb_state.last_cond_rel
                    rel_rescaled = fb_state.last_cond_rescaled if fb_state.last_cond_rescaled is not None else rel
                else:
                    prev = fb_branch.prev_sig
                    # Relative change with epsilon to avoid division by zero.
                    rel = abs(cur_sig - prev) / (abs(prev) + 1e-8)
                    # Optional EMA smoothing to reduce jitter on long videos.
                    if fb_state.ema > 0.0:
                        if fb_branch.ema_val is None:
                            fb_branch.ema_val = rel
                        else:
                            fb_branch.ema_val = fb_state.ema * fb_branch.ema_val + (1.0 - fb_state.ema) * rel
                        rel = fb_branch.ema_val
                    # Rescale according to policy (linear identity by default).
                    rel_rescaled = rescale_metric(rel, 'linear')
                # Fail-safe: invalid numbers trigger forced compute to preserve correctness.
                if not (math.isfinite(rel)):
                    fb_state.failsafe_count += 1
                    force_compute = True
            else:
                # If we cannot compute a meaningful rel (e.g., warmup), ensure compute path.
                rel = 0.0
                rel_rescaled = 0.0

            # Accumulate and decide skipping only if not forced.
            skip = False
            if not force_compute and fb_branch.prev_sig is not None:
                fb_branch.accum += float(rel_rescaled)
                fb_branch.sum_rel += float(rel)
                fb_branch.sum_rescaled += float(rel_rescaled)
                fb_branch.count_rel += 1
                skip = fb_branch.accum < float(fb_state.thresh)

            if skip and (fb_branch.prev_residual is not None):
                # Skip compute: apply cached residual to current hidden state after ensuring device/dtype match.
                res = fb_move_residual_to(fb_branch.prev_residual, x.device, x.dtype)
                if (fb_branch.shape is not None and tuple(res.shape) == tuple(fb_branch.shape)):
                    x = x + res
                    fb_branch.skipped += 1
                else:
                    # Shape/dtype mismatch: treat as anomaly and fall back to compute path.
                    fb_state.failsafe_count += 1
                    x_before = x
                    # Compute all blocks to refresh residual; if we already computed block0 for metric,
                    # reuse it to avoid double compute by starting from block 1.
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
                # Compute path: run the full stack (reusing block0 output if we computed it for residual metric),
                # cache the full-stack residual, and reset accumulation.
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

            # Update signature for next step and, if cond branch, store rel for optional CFG reuse.
            fb_branch.prev_sig = float(cur_sig)
            if fb_state.branch == 'cond':
                fb_state.last_cond_rel = float(rel)
                fb_state.last_cond_rescaled = float(rel_rescaled)
            fb_branch.total += 1

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        TRAINING-ONLY: Parameter initialization used when training from scratch.
        In inference, weights are loaded from checkpoints; this method is unused.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
