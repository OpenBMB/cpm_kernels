import torch

from ..kernels import layernorm_forward_mv, layernorm_forward_v, layernorm_backward_mv, layernorm_backward_v, layernorm_forward, inplace_mul_backward, inplace_add_backward
from .inplace import inplace_mul_add, inplace_mul, inplace_mul_addTH, inplace_mulTH

class OpLayerNormMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x : torch.Tensor, eps : float, weight : torch.Tensor, bias : torch.Tensor):
        assert x.is_cuda and x.is_contiguous() and x.ndim == 3 and x.dtype == torch.float16
        assert weight.is_cuda and weight.is_contiguous() and weight.ndim == 1 and weight.dtype == torch.float16
        out = torch.empty((x.size(0), x.size(1), x.size(2)), device=x.device, dtype=torch.float16)
        layernorm_forward(
            x.size(0), x.size(1), x.size(2),
            x.data_ptr(),
            out.data_ptr(),
            eps,
            True,
            torch.cuda.current_stream().cuda_stream
        )
        ctx.save_for_backward(x, weight)
        inplace_mul_add(out, weight, bias)
        ctx.eps = eps
        return out
    
    @staticmethod
    def backward(ctx, grad_output : torch.Tensor):
        assert grad_output.is_cuda and grad_output.is_contiguous() and grad_output.ndim == 3 and grad_output.dtype == torch.float16
        x, weight = ctx.saved_tensors
        
        mean = torch.empty((x.size(0), x.size(2)), device=x.device, dtype=torch.float16)
        var =  torch.empty((x.size(0), x.size(2)), device=x.device, dtype=torch.float16)
        layer_out = torch.empty((x.size(0), x.size(1), x.size(2)), device=x.device, dtype=torch.float16)
        layernorm_forward_mv(
            x.size(0), x.size(1), x.size(2),
            x.data_ptr(),
            layer_out.data_ptr(),
            mean.data_ptr(),
            var.data_ptr(),
            ctx.eps,
            torch.cuda.current_stream().cuda_stream
        )

        grad_bias = torch.empty((x.size(1),), device=x.device, dtype=torch.float16)
        inplace_add_backward(
            x.size(0), x.size(1), x.size(2),
            grad_output.data_ptr(),
            grad_bias.data_ptr(),
            torch.cuda.current_stream().cuda_stream
        )

        grad_weight = torch.empty((x.size(1),), device=x.device, dtype=torch.float16)
        inplace_mul_backward(
            x.size(0), x.size(1), x.size(2),
            layer_out.data_ptr(),
            grad_output.data_ptr(),
            grad_weight.data_ptr(),
            torch.cuda.current_stream().cuda_stream
        )

        inplace_mul(
            grad_output,
            weight
        )

        grad = torch.empty_like(x)
        layernorm_backward_mv(
            x.size(0), x.size(1), x.size(2),
            x.data_ptr(),
            grad_output.data_ptr(),
            mean.data_ptr(),
            var.data_ptr(),
            grad.data_ptr(),
            torch.cuda.current_stream().cuda_stream
        )
        return grad, None, grad_weight, grad_bias

class OpLayerNormNoMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x : torch.Tensor, eps : float, weight : torch.Tensor):
        assert x.is_cuda and x.is_contiguous() and x.ndim == 3 and x.dtype == torch.float16
        assert weight.is_cuda and weight.is_contiguous() and weight.ndim == 1 and weight.dtype == torch.float16
        out = torch.empty((x.size(0), x.size(1), x.size(2)), device=x.device, dtype=torch.float16)
        layernorm_forward(
            x.size(0), x.size(1), x.size(2),
            x.data_ptr(),
            out.data_ptr(),
            eps,
            False,
            torch.cuda.current_stream().cuda_stream
        )
        ctx.save_for_backward(x, weight)
        inplace_mul(out, weight)
        ctx.eps = eps
        return out
    
    @staticmethod
    def backward(ctx, grad_output : torch.Tensor):
        assert grad_output.is_cuda and grad_output.is_contiguous() and grad_output.ndim == 3 and grad_output.dtype == torch.float16
        x, weight = ctx.saved_tensors

        layer_out = torch.empty((x.size(0), x.size(1), x.size(2)), device=x.device, dtype=torch.float16)
        var =  torch.empty((x.size(0), x.size(2)), device=x.device, dtype=torch.float16)
        layernorm_forward_v(
            x.size(0), x.size(1), x.size(2),
            x.data_ptr(),
            layer_out.data_ptr(),
            var.data_ptr(),
            ctx.eps,
            torch.cuda.current_stream().cuda_stream
        )

        grad_weight = torch.empty((x.size(1),), device=x.device, dtype=torch.float16)
        inplace_mul_backward(
            x.size(0), x.size(1), x.size(2),
            layer_out.data_ptr(),
            grad_output.data_ptr(),
            grad_weight.data_ptr(),
            torch.cuda.current_stream().cuda_stream
        )

        inplace_mul(
            grad_output,
            weight
        )

        grad = torch.empty_like(x)
        layernorm_backward_v(
            x.size(0), x.size(1), x.size(2),
            x.data_ptr(),
            grad_output.data_ptr(),
            var.data_ptr(),
            grad.data_ptr(),
            torch.cuda.current_stream().cuda_stream
        )
        return grad, None, grad_weight

class LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size : int, eps : float = 1e-5, bias=True):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size)) if bias else None
    
    def forward(self, x : torch.Tensor):
        assert x.is_cuda and x.is_contiguous() and x.ndim == 3 and x.dtype == torch.float16
        assert x.size(1) == self.weight.size(0)
        
        if self.bias is not None:
            return OpLayerNormMean.apply(x, self.eps, self.weight, self.bias)
        else:
            return OpLayerNormNoMean.apply(x, self.eps, self.weight)

class LayerNormTH(torch.nn.Module):
    def __init__(self, hidden_size : int, eps : float = 1e-5, bias=True):
        super(LayerNormTH, self).__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size)) if bias else None
    
    def forward(self, x : torch.Tensor):
        old_dtype = x.dtype
        x = x.to(torch.float32)
        var = (x**2).mean(axis=1, keepdim=True)
        if self.bias is not None:
            mean = x.mean(axis=1, keepdim=True)
            x = (x - mean) * torch.rsqrt(var + self.eps)
        else:
            x = x * torch.rsqrt(var + self.eps)
        if self.bias is not None:
            x = inplace_mul_addTH(x, self.weight, self.bias)
        else:
            x = inplace_mulTH(x, self.weight)
        x = x.to(old_dtype)
        return x
