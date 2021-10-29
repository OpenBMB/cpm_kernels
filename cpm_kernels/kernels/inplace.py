from .base import Kernel, DevicePointer, CUDAStream, round_up
import ctypes

inplace_add_kernel = Kernel(
    "inplace_add",
    [
        "cu_inplace_add",
    ]
)

inplace_mask_kernel = Kernel(
    "inplace_mask",
    [
        "cu_inplace_mask"
    ]
)

inplace_mul_kernel = Kernel(
    "inplace_mul",
    [
        "cu_inplace_mul_add",
        "cu_inplace_mul",
        "cu_inplace_div",
        "cu_inplace_sub_div",
        "cu_inplace_mul_backward",
        "cu_inplace_add_backward"
    ]
)

def inplace_add(
        batch : int, n : int,
        x : DevicePointer,    # (batch, n)
        y : DevicePointer,    # (batch, n)
        stream : CUDAStream
    ):
    """
    x += y
    """
    gridDim = (batch, 1, 1)
    blockDim = (min(n, 1024), 1, 1)
    inplace_add_kernel.cu_inplace_add(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(batch),
            ctypes.c_int32(n),
            ctypes.c_void_p(x),
            ctypes.c_void_p(y)
        ]
    )

def inplace_mask(
        batch : int, n : int,
        inp : DevicePointer,    # (batch, n)
        mask : DevicePointer,   # (batch, n)
        value : float,
        stream : CUDAStream
    ):
    """
    mask
    """
    gridDim = (batch, 1, 1)
    blockDim = (min(n, 1024), 1, 1)
    inplace_mask_kernel.cu_inplace_mask(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(batch),
            ctypes.c_int32(n),
            ctypes.c_void_p(inp),
            ctypes.c_void_p(mask),
            ctypes.c_float(value)
        ]
    )

def inplace_mul_add(
        batch : int, n : int, m : int,
        inp : DevicePointer,    # (batch, n, m) fp16
        alpha : DevicePointer,  # (n)           fp16
        beta : DevicePointer,   # (n)           fp16
        stream : CUDAStream
    ):
    gridDim = (batch, n, 1)
    blockDim = (min(m, 1024), 1, 1)
    inplace_mul_kernel.cu_inplace_mul_add(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(batch),
            ctypes.c_int32(n),
            ctypes.c_int32(m),
            ctypes.c_void_p(inp),
            ctypes.c_void_p(alpha),
            ctypes.c_void_p(beta)
        ]
    )

def inplace_mul(
        batch : int, n : int, m : int,
        inp : DevicePointer,    # (batch, n, m) fp16
        alpha : DevicePointer,  # (n)           fp16
        stream : CUDAStream
    ):
    gridDim = (batch, n, 1)
    blockDim = (min(m, 1024), 1, 1)
    inplace_mul_kernel.cu_inplace_mul(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(batch),
            ctypes.c_int32(n),
            ctypes.c_int32(m),
            ctypes.c_void_p(inp),
            ctypes.c_void_p(alpha),
        ]
    )


def inplace_sub_div(
        batch : int, n : int, m : int,
        inp : DevicePointer,    # (batch, n, m) fp16
        alpha : DevicePointer,  # (n)           fp16
        beta : DevicePointer,   # (n)           fp16
        stream : CUDAStream
    ):
    gridDim = (batch, n, 1)
    blockDim = (min(m, 1024), 1, 1)
    inplace_mul_kernel.cu_inplace_sub_div(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(batch),
            ctypes.c_int32(n),
            ctypes.c_int32(m),
            ctypes.c_void_p(inp),
            ctypes.c_void_p(alpha),
            ctypes.c_void_p(beta)
        ]
    )

def inplace_div(
        batch : int, n : int, m : int,
        inp : DevicePointer,    # (batch, n, m) fp16
        alpha : DevicePointer,  # (n)           fp16
        stream : CUDAStream
    ):
    gridDim = (batch, n, 1)
    blockDim = (min(m, 1024), 1, 1)
    inplace_mul_kernel.cu_inplace_div(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(batch),
            ctypes.c_int32(n),
            ctypes.c_int32(m),
            ctypes.c_void_p(inp),
            ctypes.c_void_p(alpha),
        ]
    )

def inplace_mul_backward(
    batch : int, n : int, m : int,
    inp : DevicePointer,        # (batch, n, m) fp16
    grad_out : DevicePointer,   # (batch, n, m) fp16
    grad : DevicePointer,       # (n) fp16
    stream : CUDAStream
):
    gridDim = (n, 1, 1)
    blockDim = (32, 32, 1)
    inplace_mul_kernel.cu_inplace_mul_backward(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(batch),
            ctypes.c_int32(n),
            ctypes.c_int32(m),
            ctypes.c_void_p(inp),
            ctypes.c_void_p(grad_out),
            ctypes.c_void_p(grad)
        ]
    )

def inplace_add_backward(
    batch : int, n : int, m : int,
    grad_out : DevicePointer,   # (batch, n, m) fp16
    grad : DevicePointer,       # (n) fp16
    stream : CUDAStream
):
    gridDim = (n, 1, 1)
    blockDim = (32, 32, 1)
    inplace_mul_kernel.cu_inplace_add_backward(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(batch),
            ctypes.c_int32(n),
            ctypes.c_int32(m),
            ctypes.c_void_p(grad_out),
            ctypes.c_void_p(grad)
        ]
    )