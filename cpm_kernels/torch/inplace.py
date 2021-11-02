import torch
from ..kernels import inplace


def inplace_mask(x : torch.Tensor, mask : torch.Tensor, value : float) -> torch.Tensor:
    assert x.is_contiguous() and x.is_cuda and x.dtype == torch.float16
    assert mask.is_contiguous() and mask.is_cuda and mask.dtype == torch.bool
    assert x.device == mask.device
    assert x.size() == mask.size()
    batch = x.size(0)
    stride = x.stride(0)
    inplace.inplace_mask(
        batch, stride,
        x.data_ptr(),
        mask.data_ptr(),
        value,
        torch.cuda.current_stream().cuda_stream
    )
    return x

def inplace_maskTH(x : torch.Tensor, mask : torch.Tensor, value : float) -> torch.Tensor:
    return torch.where(
        mask,
        x,
        torch.scalar_tensor(value, device=x.device, dtype=x.dtype),
    )

