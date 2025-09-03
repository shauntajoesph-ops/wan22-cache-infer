import pytest
import torch

from wan.utils.teacache import move_residual_to as tc_move
from wan.utils.fbcache import move_residual_to as fb_move


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for device move test")
def test_teacache_move_cpu_to_gpu_and_back():
    # Create a CPU tensor and move to GPU, then back to CPU using helper
    cpu_t = torch.ones(4, 5, dtype=torch.float32, device=torch.device("cpu"))
    gpu_t = tc_move(cpu_t, torch.device("cuda:0"), torch.float16)
    assert gpu_t.device.type == 'cuda' and gpu_t.dtype == torch.float16
    # Move back to CPU and original dtype
    cpu_t2 = tc_move(gpu_t, torch.device("cpu"), torch.float32)
    assert cpu_t2.device.type == 'cpu' and cpu_t2.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for device move test")
def test_fbcache_move_cpu_to_gpu_and_back():
    # Same test for FBCache helper
    cpu_t = torch.ones(2, 3, dtype=torch.bfloat16, device=torch.device("cpu"))
    gpu_t = fb_move(cpu_t, torch.device("cuda:0"), torch.float16)
    assert gpu_t.device.type == 'cuda' and gpu_t.dtype == torch.float16
    cpu_t2 = fb_move(gpu_t, torch.device("cpu"), torch.bfloat16)
    assert cpu_t2.device.type == 'cpu' and cpu_t2.dtype == torch.bfloat16

