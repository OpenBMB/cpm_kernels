from .base import Kernel, DevicePointer, CUDAStream, round_up
import ctypes

gemv_kernel = Kernel(
    "gemv",
    [
        "cu_gemv_broadcast_mat_int8",
        "cu_gemv_fp16",
        "cu_gemv_fp16_transpose",
        "cu_gemv_broadcast_mat_fp16"
    ]
)

def gemv_broadcast_mat_int8(
    batch : int, dim_out : int, dim_in : int,
    scale_mat : DevicePointer,          # (dim_out,)    fp16
    mat : DevicePointer,                # (dim_out, dim_in) int8
    vec : DevicePointer,                # (batch, dim_in)   fp16
    out : DevicePointer,                # (batch, dim_out)  fp16
    stream : CUDAStream
):
    assert dim_in % 4 == 0
    gridDim = (batch, 1, 1)
    blockDim = (min(1024, round_up(dim_in // 4, 32)), 1, 1)
    gemv_kernel.cu_gemv_broadcast_mat_int8(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(batch),
            ctypes.c_int32(dim_out),
            ctypes.c_int32(dim_in),
            ctypes.c_void_p(scale_mat),
            ctypes.c_void_p(mat),
            ctypes.c_void_p(vec),
            ctypes.c_void_p(out)
        ]
    )

def gemv_fp16(
    batch : int, dim_out : int, dim_in : int,
    mat : DevicePointer,                # (batch, dim_out, dim_in) fp16
    vec : DevicePointer,                # (batch, dim_in)   fp16
    out : DevicePointer,                # (batch, dim_out)  fp16
    stream : CUDAStream
):
    assert dim_in % 2 == 0
    gridDim = (batch, 1, 1)
    blockDim = (min(1024, round_up(dim_in // 2, 32)), 1, 1)
    gemv_kernel.cu_gemv_fp16(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(batch),
            ctypes.c_int32(dim_out),
            ctypes.c_int32(dim_in),
            ctypes.c_void_p(mat),
            ctypes.c_void_p(vec),
            ctypes.c_void_p(out)
        ]
    )


def gemv_fp16_transpose(
    batch : int, dim_out : int, dim_in : int,
    mat : DevicePointer,                # (batch, dim_in, dim_out) fp16
    vec : DevicePointer,                # (batch, dim_in)   fp16
    out : DevicePointer,                # (batch, dim_out)  fp16
    stream : CUDAStream
):
    gridDim = (batch, round_up(dim_out, 32) // 32, 1)
    blockDim = (32, 32, 1)
    gemv_kernel.cu_gemv_fp16_transpose(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(batch),
            ctypes.c_int32(dim_out),
            ctypes.c_int32(dim_in),
            ctypes.c_void_p(mat),
            ctypes.c_void_p(vec),
            ctypes.c_void_p(out)
        ]
    )

def gemv_broadcast_mat_fp16(
    batch : int, dim_out : int, dim_in : int,
    mat : DevicePointer,                # (dim_out, dim_in) fp16
    vec : DevicePointer,                # (batch, dim_in)
    out : DevicePointer,                # (batch, dim_out)
    stream : CUDAStream
):
    assert dim_in % 2 == 0
    gridDim = (batch, 1, 1)
    blockDim = (min(1024, round_up(dim_in // 2, 32)), 1, 1)
    gemv_kernel.cu_gemv_broadcast_mat_fp16(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(batch),
            ctypes.c_int32(dim_out),
            ctypes.c_int32(dim_in),
            ctypes.c_void_p(mat),
            ctypes.c_void_p(vec),
            ctypes.c_void_p(out)
        ]
    )