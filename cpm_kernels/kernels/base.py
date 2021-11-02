import ctypes
import os
from typing import List, Any, Tuple
from ..library import cuda, cudart
from ..device import Device, num_devices
import pkg_resources
DevicePointer = int
CUDAStream = cudart.cudaStream_t

RESOURCE_PACKAGE_NAME = __name__

def round_up(x : int, m : int) -> int:
    return (x + m - 1) // m * m

class KernelFunction:
    def __init__(self, funcs : List[cuda.CUfunction]) -> None:
        self._funcs = funcs
    
    def __call__(self, gridDim : Tuple[int, int, int], blockDim : Tuple[int, int, int], 
            sharedMemBytes : int, stream : cudart.cudaStream_t, params : List[Any] ) -> None:
        curr_device = cudart.cudaGetDevice()
        func = self._funcs[curr_device]
        cudart.cudaSetDevice(curr_device)   # pytorch ctx changed
        cuda.cuLaunchKernel(func, 
            gridDim[0], gridDim[1], gridDim[2], 
            blockDim[0], blockDim[1], blockDim[2], 
            sharedMemBytes, stream, [
                ctypes.addressof(p) for p in params
            ]
        )
        

class Kernel:
    def __init__(self, filename : str, function_names : List[str]):
        filename = filename + ".fatbin"
        filename = os.path.join("cuda", filename)
        if not pkg_resources.resource_exists(RESOURCE_PACKAGE_NAME, filename):
            raise RuntimeError("File `%s` not found in `%s`" % (filename, RESOURCE_PACKAGE_NAME))
        self.filename = filename
        self.code = pkg_resources.resource_string(RESOURCE_PACKAGE_NAME, filename)
        self._function_names = function_names

        
        curr_device = cudart.cudaGetDevice()
        funcs = {
            name: [] for name in self._function_names
        }

        for idx in range(num_devices()):
            Device(idx).use()
            cumodule = cuda.cuModuleLoadData(self.code)
            for name in self._function_names:
                func = cuda.cuModuleGetFunction(cumodule, name)
                funcs[name].append(func)
        cudart.cudaSetDevice(curr_device)
        
        for name in self._function_names:
            setattr(self, name, KernelFunction(funcs[name]))






