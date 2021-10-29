from .base import Kernel, DevicePointer, CUDAStream
import ctypes

embedding_kernel = Kernel(
    "position_bucket",
    [
        "cu_position_bucket",
    ]
)


def position_bucket(
        query_len : int, 
        key_len : int, 
        num_buckets : int, 
        max_distance : int, 
        out : DevicePointer,    # (key_len, query_len)
        bidirectional : bool, 
        stream : CUDAStream
    ) -> None:

    gridDim = (key_len, 1, 1)
    blockDim = (min(query_len, 1024), 1, 1)
    embedding_kernel.cu_position_bucket(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(query_len),
            ctypes.c_int32(key_len),
            ctypes.c_int32(num_buckets),
            ctypes.c_int32(max_distance),
            ctypes.c_void_p(out),
            ctypes.c_bool(bidirectional)
        ]
    )