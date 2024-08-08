import torch
from torchao.dtypes import AffineQuantizedTensor

from .utils import is_available


aten = torch.ops.aten


def _get_to_kwargs(self: torch.nn.Module, *args, **kwargs):
    device, dtype, _, memory_format = torch._C._nn._parse_to(*args, **kwargs)
    device = self.device if device is None else device
    dtype = self.dtype if dtype is None else dtype
    memory_format = (
        memory_format if memory_format is not None else torch.preserve_format
    )
    kwargs = {
        "device": device,
        "dtype": dtype,
        "memory_format": memory_format,
    }
    return kwargs


def apply_fixes():
    if is_available("torchao.prototype.quant_llm"):
        from torchao.prototype.quant_llm import QuantLlmLinearWeight

        # Stinky fix for aten::to::dtype_layout on QuantLlmLinearWeight
        QuantLlmLinearWeight.to = move_fpx

        # Even stinkier fix for aten::_has[...] on QuantLlmLinearWeight
        @QuantLlmLinearWeight.implements(aten._has_compatible_shallow_copy_type.default)
        def _(f, types, *args, **kwargs):
            return False

    @AffineQuantizedTensor.implements(aten._has_compatible_shallow_copy_type.default)
    def _(f, types, *args, **kwargs):
        return False


def move_fpx(self, *args, **kwargs):
    kwargs = _get_to_kwargs(self, *args, **kwargs)

    device = kwargs.pop("device")
    kwargs.pop("memory_format")  # unsupported

    return self.__class__(
        self.fpx_data.to(device=device),
        self.scale.to(device=device),
        self.ebits,
        self.mbits,
    )
