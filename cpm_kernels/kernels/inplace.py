from .base import Kernel, DevicePointer, CUDAStream, round_up
import ctypes


inplace_mask_kernel = Kernel(
    "inplace_mask",
    [
        "cu_inplace_mask"
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
