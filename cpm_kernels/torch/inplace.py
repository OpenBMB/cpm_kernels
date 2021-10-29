import torch
from ..kernels import inplace


def inplace_add(x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous() and x.is_cuda and x.dtype == torch.float16
    assert y.is_contiguous() and y.is_cuda and y.dtype == torch.float16
    assert x.device == y.device
    batch = x.size(0)
    stride = x.stride(0)
    inplace.inplace_add(
        batch, stride,
        x.data_ptr(),
        y.data_ptr(),
        torch.cuda.current_stream().cuda_stream
    )
    return x

def inplace_mask(x : torch.Tensor, mask : torch.Tensor, value : float) -> torch.Tensor:
    assert x.is_contiguous() and x.is_cuda and x.dtype == torch.float16
    assert mask.is_contiguous() and mask.is_cuda and mask.dtype == torch.bool
    assert x.device == mask.device
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

def inplace_mul(x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous() and x.is_cuda and x.dtype == torch.float16
    assert y.is_contiguous() and y.is_cuda and y.dtype == torch.float16
    assert x.ndim == 3 and y.ndim == 1
    assert x.device == y.device
    batch, n, m = x.size()
    assert y.size(0) == n

    inplace.inplace_mul(
        batch, n, m,
        x.data_ptr(),
        y.data_ptr(),
        torch.cuda.current_stream().cuda_stream
    )
    return x

def inplace_mul_add(x : torch.Tensor, y : torch.Tensor, z : torch.Tensor) -> torch.Tensor:
    """
    x = x * y + z
    """
    assert x.is_contiguous() and x.is_cuda and x.dtype == torch.float16
    assert y.is_contiguous() and y.is_cuda and y.dtype == torch.float16
    assert z.is_contiguous() and z.is_cuda and z.dtype == torch.float16
    assert x.ndim == 3 and y.ndim == 1 and z.ndim == 1
    assert x.device == y.device and x.device == z.device
    batch, n, m = x.size()
    assert y.size(0) == n and z.size(0) == n

    inplace.inplace_mul_add(
        batch, n, m,
        x.data_ptr(),
        y.data_ptr(),
        z.data_ptr(),
        torch.cuda.current_stream().cuda_stream
    )
    return x

def inplace_div(x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous() and x.is_cuda and x.dtype == torch.float16
    assert y.is_contiguous() and y.is_cuda and y.dtype == torch.float16
    assert x.ndim == 3 and y.ndim == 1
    assert x.device == y.device
    batch, n, m = x.size()
    assert y.size(0) == n

    inplace.inplace_div(
        batch, n, m,
        x.data_ptr(),
        y.data_ptr(),
        torch.cuda.current_stream().cuda_stream
    )
    return x

def inplace_sub_div(x : torch.Tensor, y : torch.Tensor, z : torch.Tensor) -> torch.Tensor:
    """
    x = (x - z) / y
    """
    assert x.is_contiguous() and x.is_cuda and x.dtype == torch.float16
    assert y.is_contiguous() and y.is_cuda and y.dtype == torch.float16
    assert z.is_contiguous() and z.is_cuda and z.dtype == torch.float16
    assert x.ndim == 3 and y.ndim == 1 and z.ndim == 1
    assert x.device == y.device and x.device == z.device
    batch, n, m = x.size()
    assert y.size(0) == n and z.size(0) == n

    inplace.inplace_sub_div(
        batch, n, m,
        x.data_ptr(),
        y.data_ptr(),
        z.data_ptr(),
        torch.cuda.current_stream().cuda_stream
    )
    return x


@torch.jit.script
def inplace_sub_divTH(x : torch.Tensor, y : torch.Tensor, z : torch.Tensor) -> torch.Tensor:
    return (x - z[None, :, None]) / y[None, :, None]

@torch.jit.script
def inplace_divTH(x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
    return x / y[None, :, None]

@torch.jit.script
def inplace_mul_addTH(x : torch.Tensor, y : torch.Tensor, z : torch.Tensor) -> torch.Tensor:
    return x * y[None, :, None] + z[None, :, None]

@torch.jit.script
def inplace_mulTH(x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
    return x * y[None, :, None]
