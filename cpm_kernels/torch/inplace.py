import torch
from ..kernels import inplace


def inplace_add(x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous() and x.is_cuda and x.dtype == torch.float16
    assert y.is_contiguous() and y.is_cuda and y.dtype == torch.float16
    assert x.device == y.device
    assert x.size() == y.size()
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

class OpBatchedAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        assert x.is_contiguous() and x.is_cuda and x.dtype == torch.float16
        assert y.is_contiguous() and y.is_cuda and y.dtype == torch.float16
        assert x.device == y.device
        assert x.size()[1:] == y.size()

        out = torch.empty(x.size(), device=x.device, dtype=x.dtype)
        inplace.batched_add_forward(
            x.size(0),
            x.stride(0),
            x.data_ptr(),
            y.data_ptr(),
            out.data_ptr(),
            torch.cuda.current_stream().cuda_stream
        )
        return out

    @staticmethod
    def backward(ctx, grad_output : torch.Tensor):
        assert grad_output.is_contiguous() and grad_output.is_cuda and grad_output.dtype == torch.float16
        grad_y = torch.empty( grad_output.size()[1:], device=grad_output.device, dtype=grad_output.dtype)
        inplace.batched_add_backward(
            grad_output.size(0),
            grad_output.stride(0),
            grad_output.data_ptr(),
            grad_y.data_ptr(),
            torch.cuda.current_stream().cuda_stream
        )
        return grad_output, grad_y

def batched_add(x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
    return OpBatchedAdd.apply(x, y)

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

def inplace_maskTH(x : torch.Tensor, mask : torch.Tensor, value : float) -> torch.Tensor:
    return torch.where(
        mask,
        x,
        torch.scalar_tensor(value, device=x.device, dtype=x.dtype),
    )

def batched_addTH(x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
    return x + y