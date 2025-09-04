"""Lightweight CUDA Graphs runner utilities (inference-only).

This module exposes a minimal helper to capture and replay a fixed-shape
forward function using torch.cuda.CUDAGraph. It is designed for steady-state
sampling loops where tensor shapes and memory layouts remain constant.

Notes and caveats:
- Only use when shapes/dtypes/devices are static and the model is not offloaded.
- Graphs capture the memory pool; avoid interleaving other allocations.
- If capture fails, callers should transparently fall back to eager execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import torch


@dataclass
class CUDAGraphRunner:
    """Capture and replay a 2-arg forward: fn(latent, timestep) -> out.

    The runner owns static clones of the input tensors. On replay, new data is
    copied into those buffers and the captured graph is executed, updating the
    output reference in-place.
    """

    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    example_latent: torch.Tensor
    example_timestep: torch.Tensor
    warmup: int = 1

    def __post_init__(self) -> None:
        assert self.example_latent.is_cuda and self.example_timestep.is_cuda
        # Static placeholders with identical shapes/dtypes/devices
        self._lat = self.example_latent.clone()
        self._ts = self.example_timestep.clone()
        self._out_ref: Optional[torch.Tensor] = None
        self._graph: Optional[torch.cuda.CUDAGraph] = None

        # Warmup eager runs to allocate parameter buffers
        for _ in range(max(0, int(self.warmup))):
            _ = self.fn(self.example_latent, self.example_timestep)

        # Capture graph on a dedicated stream
        self._graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(self._graph):
            self._out_ref = self.fn(self._lat, self._ts)

    def replay(self, new_latent: torch.Tensor, new_timestep: torch.Tensor) -> torch.Tensor:
        assert self._graph is not None and self._out_ref is not None
        # Copy new data into static buffers (no shape changes allowed)
        self._lat.copy_(new_latent)
        self._ts.copy_(new_timestep)
        self._graph.replay()
        return self._out_ref

