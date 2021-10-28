from .base import Kernel, DevicePointer, CUDAStream, round_up
import ctypes

gelu_kernel = Kernel(
    "gemm",
    [
        "cu_gemm_round",
        "cu_gemm_round_transpose",
        "cu_gemm_scale",
        "cu_gemm_calc_scale",
        "cu_gemm_calc_scale_transpose"
    ]
)

def gemm_round(
        batch : int,
        n : int,
        m : int,
        mat : DevicePointer,    # (b, n, m)
        scale : DevicePointer,  # (b, n)
        out : DevicePointer,    # (b, n, m)
        stream : CUDAStream
    ):
    gridDim = (batch, n, 1)
    blockDim = (min(m, 1024), 1, 1)
    gelu_kernel.cu_gemm_round(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(batch),
            ctypes.c_int32(n),
            ctypes.c_int32(m),
            ctypes.c_void_p(mat),
            ctypes.c_void_p(scale),
            ctypes.c_void_p(out)
        ]
    )

def gemm_round_transpose(
        batch : int,
        n : int,
        m : int,
        mat : DevicePointer,    # (b, n, m)
        scale : DevicePointer,  # (b, n)
        out : DevicePointer,    # (b, n, m)
        stream : CUDAStream
    ):
    gridDim = (batch, n, 1)
    blockDim = (min(m, 1024), 1, 1)
    gelu_kernel.cu_gemm_round_transpose(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(batch),
            ctypes.c_int32(n),
            ctypes.c_int32(m),
            ctypes.c_void_p(mat),
            ctypes.c_void_p(scale),
            ctypes.c_void_p(out)
        ]
    )

def gemm_scale(
        batch : int, n : int, m : int,
        mat : DevicePointer,        # (b, n, m)
        scale_x : DevicePointer,    # (b, n),
        scale_y : DevicePointer,    # (b, m),
        out : DevicePointer,        # (b, n, m)
        broad_cast_x : bool,
        broad_cast_y : bool,
        stream : CUDAStream
    ):
    gridDim = (batch, n, 1)
    blockDim = (min(m, 1024), 1, 1)
    gelu_kernel.cu_gemm_scale(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(batch),
            ctypes.c_int32(n),
            ctypes.c_int32(m),
            ctypes.c_void_p(mat),
            ctypes.c_void_p(scale_x),
            ctypes.c_void_p(scale_y),
            ctypes.c_void_p(out),
            ctypes.c_bool(broad_cast_x),
            ctypes.c_bool(broad_cast_y)
        ]
    )

def gemm_calc_scale(
        batch : int, n : int, m : int,
        mat : DevicePointer,    # (b, n, m)
        out : DevicePointer,    # (b, n)
        stream : CUDAStream
    ):
    gridDim = (batch, n, 1)
    blockDim = (min(round_up(m, 32), 1024), 1, 1)
    gelu_kernel.cu_gemm_calc_scale(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(batch),
            ctypes.c_int32(n),
            ctypes.c_int32(m),
            ctypes.c_void_p(mat),
            ctypes.c_void_p(out)
        ]
    )

def gemm_calc_scale_transpose(
        batch : int, n : int, m : int,
        inp : DevicePointer,    # (b, n, m)
        out : DevicePointer,    # (b, m)
        stream : CUDAStream
    ):
    gridDim = (batch, round_up(m, 32) // 32, 1)
    blockDim = (32, 32, 1)
    gelu_kernel.cu_gemm_calc_scale_transpose(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(batch),
            ctypes.c_int32(n),
            ctypes.c_int32(m),
            ctypes.c_void_p(inp),
            ctypes.c_void_p(out)
        ]
    )