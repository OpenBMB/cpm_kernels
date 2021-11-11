from ..library import cublaslt
from .base import Kernel, DevicePointer, CUDAStream, round_up
import ctypes

utils_kernel = Kernel(
    "utils",
    [
        "copy_data_to_kv"
    ]
)


def copy_data_to_kv(
    batch : int, buffer_len : int, n : int,
    inp : DevicePointer,        # (batch, n)
    out : DevicePointer,        # (batch, buffer_len, n)
    pos : int,
    stream : CUDAStream
):
    assert n % 2 == 0
    gridDim = (batch, 1, 1)
    blockDim = (min(1024, n // 2), 1, 1)
    utils_kernel.copy_data_to_kv(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(batch),
            ctypes.c_int32(buffer_len),
            ctypes.c_int32(n),
            ctypes.c_void_p(inp),
            ctypes.c_void_p(out),
            ctypes.c_int32(pos)
        ]
    )

