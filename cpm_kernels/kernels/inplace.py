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

def inplace_add(
        batch : int, n : int,
        inp : DevicePointer,    # (batch, n)
        out : DevicePointer,    # (batch, n)
        stream : CUDAStream
    ):
    gridDim = (batch, 1, 1)
    blockDim = (min(n, 1024), 1, 1)
    inplace_add_kernel.cu_inplace_add(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(batch),
            ctypes.c_int32(n),
            ctypes.c_void_p(inp),
            ctypes.c_void_p(out)
        ]
    )

def inplace_mask(
        batch : int, n : int,
        inp : DevicePointer,    # (batch, n)
        mask : DevicePointer,   # (batch, n)
        out : DevicePointer,    # (batch, n)
        stream : CUDAStream
    ):
    gridDim = (batch, 1, 1)
    blockDim = (min(n, 1024), 1, 1)
    inplace_mask_kernel.cu_inplace_mask(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(batch),
            ctypes.c_int32(n),
            ctypes.c_void_p(inp),
            ctypes.c_void_p(mask),
            ctypes.c_void_p(out)
        ]
    )