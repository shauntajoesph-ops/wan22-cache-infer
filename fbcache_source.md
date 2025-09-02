1) What FB Cache is (and why it works)

Idea (informal): At many denoising steps, the early Transformer block(s) change very little. If we can detect when the first block’s output/residual hasn’t changed “enough” compared to a recent step, we can skip all remaining blocks and reuse a cached residual/update instead of recomputing the full stack.
Community docs expose this via a residual difference threshold (e.g., residual_diff_threshold); higher thresholds yield more skipping (faster) but risk quality loss. Typical reported settings for image models are around 0.08–0.12 (≈1.5–2.0×), with model‑/step‑count‑specific variation. 
Nunchaku
GitHub
RunningHub

What to compare: Most implementations measure the L1/L2 norm difference between the first Transformer block output/residual at step t vs that from a previous step (often t‑1). If the relative difference is below a threshold, reuse; else recompute. ParaAttention’s docs explicitly describe using the first‑block residual difference as a proxy for overall model output change and report up to ~2× speedups on HunyuanVideo with good quality. 
GitHub
DeepWiki

Relation to Delta‑DiT: Δ‑DiT proposes Δ‑Cache, caching incremental offsets across blocks to mitigate quality drop when skipping. You can combine first‑block gating (cheap decision) with delta‑style residuals (better reconstruction) for a more conservative skip. 
arXiv
OpenReview

2) Common design variants in the wild

“Indicator‑only” FBCache (most common)

Use the first block residual difference as a skip/compute gate, but the reused payload is the cached full‑stack residual (output − input before block stack), similar to TeaCache.

Pros: maximal skip (skip the entire stack), minimal plumbing.

Cons: quality may drift if gating is too permissive.
Evidence: ParaAttention & Comfy‑WaveSpeed expositions; Nunchaku’s apply_cache_on_pipe(..., residual_diff_threshold=0.12) recommends ~2× at 50 steps, ~1.4× at 30 steps. 
GitHub
Nunchaku

“First‑block reuse” FBCache (safer, lower speedup)

Reuse only the first block’s output; compute all subsequent blocks as usual.

Pros: safer visual fidelity; Cons: less speedup (you still pay for many blocks).

Δ‑style FBCache (hybrid)

Keep first‑block gating, but the reused payload is a set of block‑wise deltas (Δ‑Cache).

Pros: better quality at aggressive skip; Cons: higher implementation complexity. 
arXiv

Policy knobs (commonly surfaced)

Threshold (residual_diff_threshold): trade speed vs quality (e.g., 0.08–0.12 often cited).

Warmup / last‑steps windows: disable caching early/late where dynamics are large.

Max consecutive hits: avoid long skip streaks in case of drift.

Start/End schedule: only enable between certain normalized step ratios (e.g., 0.2–0.8). Community issues note batch‑size and scheduling can affect stability. 
GitHub
+1

3) Where to tap the signal (ground truth, not approximation)

Compute the exact tensor that the first self‑attention block actually produces (post‑norm + modulation; in DiT this is the tensor passed into self‑attn after the first AdaLN/Norm).

From this, compute either:

the residual w.r.t. the pre‑block tensor and compare across steps, or

the block output itself (consistent dtype/shape).

Several toolkits expose this via turnkey functions or patches in ComfyUI / Diffusers pipelines. 
GitHub
+1
ComfyAI

4) Step‑by‑step blueprint to implement a robust FBCache

The steps below are implementation‑agnostic and work for DiT‑style image & video pipelines; they also compose with TeaCache, context/sequence parallelism, offload, and CFG.

4.1 State & API surface (per model instance)

State (per branch for CFG):
prev_sig (CPU scalar summary of first‑block output/residual),
prev_stack_residual (x_after − x_before over the full block stack),
accum (optional running score),
counters: hits, forced, cnt.

Config/flags:
enabled, threshold, metric (L1/L2/relative), warmup, last_steps, max_consecutive_hits, window=[start,end], per_frame (video), per_expert (multi‑expert).

APIs: set_branch(cond|uncond), reset(), maybe_skip(x, step_idx, meta) → {skip|compute, reason}, update_after_compute(x_before, x_after).
(Nunchaku and ParaAttention both expose “apply first block cache” adapters in pipelines; you’ll mirror that idea.) 
Nunchaku
Hugging Face

4.2 The gating decision

Compute the first‑block tensor (post‑norm/mod) deterministically on every call—even when skipping—so the gate is based on current input.

Summarize to a stable scalar signature (e.g., mean absolute per‑token diff folded to one scalar) and detach to CPU; maintain a small eps in denominators.

Check fast disablers: warmup, last steps, new shape/frame/expert, branch switch → force compute and reset(accum).

Metric: rel = ||sig − prev_sig|| / (|prev_sig| + eps) (or direct L2/L1 on the full tensor if you choose; scalar signatures reduce overhead).

Schedule & policy: enforce [start,end] window; optionally add accumulator and max_consecutive_hits.

Decision: skip = (rel < threshold) (or accum < threshold if accumulating).

Community defaults: ~0.12 often recommended for FLUX 50‑step (≈2×), lower for fewer steps (e.g., ~1.4× at 30). Video models tend to require more conservative thresholds. 
Nunchaku

4.3 The skip path vs compute path

Compute path (refresh):
Save x_before at block‑stack entry → run all blocks → set prev_stack_residual = (x_after − x_before).detach() (store on CPU if offloading) → prev_sig = cur_sig.

Skip path (reuse):
Ensure prev_stack_residual is on the active device/dtype → do x = x + prev_stack_residual → update counters → do not run blocks.

Variant: “first‑block reuse” path: reuse first‑block output and continue; smaller speedup, higher fidelity.

Hybrid (Δ‑style): if you’ve stored per‑block deltas (heavier), you can reconstruct a closer approximation when skipping. 
arXiv

5) Interactions & lifecycle
5.1 Classifier‑Free Guidance (CFG)

Maintain separate caches per branch (cond/uncond).

Increment step counters on cond branch only (to align with one “denoising step” = {uncond, cond}).

If you also use a CFG cache (within‑step cond/uncond batching/reuse), keep it orthogonal: FBCache is across steps; CFG cache is within a step. (Related CFG research for context only. ) 
arXiv
+1

5.2 TeaCache co‑existence

TeaCache: gate using first‑block modulated input and reuse stack residual across steps. FBCache: gate using first‑block output/residual; reuse may be identical (stack residual).

Composition strategy:

Use FBCache as the primary gate (it “looks” at the earliest, easiest signal).

Optionally AND it with TeaCache’s gate (more conservative), or OR it (more aggressive).

Keep shared payload (prev_stack_residual) or separate payloads if you want to gather telemetry per method.

Most users prefer one of them at a time; community comparisons suggest either TeaCache or FBCache is enough for images; video may benefit from more conservative policies. 
Sahirp

5.3 Video specifics

For frame‑major VAE layouts you can maintain a per‑frame signature, but for joint spatio‑temporal DiTs it’s safer to maintain one gate for the whole token grid (or one per segment in segmented motion tokens).

Reset signatures when latent window or resolution changes (long‑video windows/overlaps). ParaAttention notes that FBCache can be combined with context parallel and quantization for significant speedups in video. 
GitHub

5.4 Distributed (context/sequence parallel)

Compute the local rel scalar on each rank → AllReduce (mean) to obtain a consistent decision → each rank applies its own sharded residual locally.

Never call collectives conditionally; guard with dist.is_initialized().

ParaAttention explicitly demonstrates Context Parallelism + FBCache composition. 
DeepWiki

5.5 Offload & memory

Store prev_stack_residual on CPU when the model is offloaded; move lazily to GPU on use.

Check dtype at addition (cast to model’s param_dtype).

5.6 Reset conditions (hard rules)

Any shape/seq‑len change (resolution, latent window, batch shape).

Expert/model switch (e.g., low‑noise/high‑noise experts).

Branch change: enforced by separate states.

NaN/Inf or dtype/device mismatch: force compute, reset, log.

6) Telemetry, knobs, and safe defaults

Expose: threshold, start, end, warmup, last_steps, max_consecutive_hits, metric, per_frame.

Log (end‑of‑run): total steps, skipped steps, skip rate, average rel, forced computes (warmup/last), average time/step.

Defaults (starting points):

Images (FLUX‑like 50 steps): threshold ≈ 0.12 (reports ≈2×); for 30 steps expect ≈1.4×. Reduce if artifacts appear. 
Nunchaku

Videos (Hunyuan‑like): start lower (e.g., 0.08–0.10) and tune up carefully. ParaAttention reports up to ~2× w/ good quality. 
GitHub

Batches >1 may require lower thresholds for stability. 
GitHub

7) Worked examples (how the blueprint plays out)
Example A — FLUX image, 50 steps, 1024×1024

Setup: FBCache enabled, threshold=0.12, warmup=1, last_steps=1, start=0.2, end=0.9, max_hits=5.

Expected: Skip rate often ~40–60%; wall‑clock ~2× per Nunchaku’s guidance. If artifacts (hands/text), drop to 0.08–0.10. 
Nunchaku

Example B — HunyuanVideo, 30 steps, context parallel ×2

Setup: FBCache + CP; threshold=0.08–0.10, per‑frame disabled (global gate), warmup=2, last_steps=2.

Expected: ~1.5–2× speedup while maintaining quality; combine with FP8/torch.compile to go higher (ParaAttention shows stacking gains). 
DeepWiki

Example C — Aggressive stack (for benchmarking, not default)

Setup: threshold=0.12 + FP8 dynamic quant + torch.compile + CP.

Observed (community table): With 1–4 GPUs, stacking FBCache + FP8 + CP yielded 1.5× → 6.7× in a published ParaAttention example (model/hardware‑dependent). 
DeepWiki

8) Pitfalls & how to avoid them

Incorrect tap point: Comparing the wrong tensor (pre‑modulation or post‑stack) makes the gate noisy. Always mirror the exact first‑block output/residual used by self‑attention. 
GitHub

Threshold too high: Visible drift or “mushy” backgrounds; reduce threshold or shrink the active window. Community reports commonly center around 0.08–0.12 for images. 
Nunchaku

Batch >1 instability: Lower the threshold; skip policies may need per‑sample signatures or conservative global gating. 
GitHub

Conditional collectives: Never call AllReduce only on some ranks. All ranks must participate every step (even if not skipping).

CFG & cache mixing: Keep branch‑separate states; do not reuse cond residual for uncond (and vice versa).

Long‑video windows: Reset when the latent frame window shifts (overlap/hop); otherwise signatures go stale.

9) How this relates to TeaCache (when to use which)

TeaCache: gate using first‑block modulated input difference; reuse stack residual; proven 1.4–2.1× ranges in community reports.

FBCache: gate using first‑block output/residual difference; reuse stack residual (indicator‑only) or Δ‑style payload.

Rule of thumb:

Images → either TeaCache or FBCache suffices (start with one).

Videos → start conservatively and consider TeaCache (less aggressive) if FBCache shows artifacts, or vice versa. Mixed strategies can be tested, but keep decisions simple. 
Sahirp

10) Minimal “DESIGN.md” skeleton you can adopt (no code)

Use this to document your FBCache spec in your repo.

Title: First‑Block Cache (FBCache) — Design

Overview & Goals — training‑free, inference‑only acceleration; quality‑aware skipping.

Scope & Non‑goals — supported pipelines/models; no training or optimizer changes.

Architecture

Tap point (first‑block output/residual), metric (relative L1/L2), threshold & window, accumulator, max_hits.

Payload reuse: stack residual (TeaCache‑like) vs Δ‑style (optional).

APIs & Flags — enable, threshold, start/end, warmup/last, max_hits, metric, per_frame.

Lifecycle — init, per‑branch/per‑expert state, offload residency, reset triggers (shape/window/expert/NaN).

Distributed — rank‑consistent decision via AllReduce(mean), sharded residual add.

CFG Interaction — separate branch states; cond‑only step counting.

Offload & Memory — CPU store for cached residual; lazy device moves; dtype checks.

Telemetry — end‑of‑run skip stats; optional per‑step debug (rate‑limited).

Defaults & Tuning — starting thresholds (e.g., images 0.12@50‑step), schedules, batch notes. 
Nunchaku

Compatibility — with FP8, torch.compile, context/sequence parallel. 
DeepWiki

Risks & Fallbacks — drift/quality, divergence, shape changes → force compute + reset.

Appendix — references to ParaAttention / Nunchaku / WaveSpeed docs & Δ‑DiT paper. 
Hugging Face
Nunchaku
GitHub
arXiv

References (key sources)

Nunchaku docs — FBCache usage & recommended thresholds/speedups (e.g., 0.12 → ~2× @50 steps, ~1.4× @30). 
Nunchaku

Comfy‑WaveSpeed — residual_diff_threshold knob; 1.5–3× speedups claim (model‑dependent). 
GitHub
+1
RunningHub

ParaAttention (DeepWiki & docs) — FBCache using first‑block residual diff; composition with Context Parallelism; performance tables & examples. 
DeepWiki
+1

Δ‑DiT paper — Δ‑Cache (cache offsets across blocks), complementary to first‑block gating; training‑free. 
arXiv

Comparative blog — “First‑Block Cache vs TeaCache vs AdaCache” (practical guidance). 
Sahirp

Caveats — batch size sensitivity: lower thresholds for batch>1. 
GitHub

ComfyUI patch node — “ApplyFirstBlockCachePatch” (what users see in practice). 
ComfyAI
