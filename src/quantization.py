from functools import partial

import torch
from torchao.dtypes import (
    PlainLayoutType,
    SemiSparseLayoutType,
    to_affine_quantized,
    TensorCoreTiledLayoutType,
)
from torchao.float8.float8_linear import Float8Tensor
from torchao.quantization import (
    quantize_,
    to_linear_activation_quantized,
)
from torchao.quantization.utils import _get_per_token_block_size
from torchao.quantization.quant_primitives import MappingType, ZeroPointDomain


def _get_linear_subclass_inserter(constructor):
    def insert_subclass(lin):
        lin.weight = torch.nn.Parameter(constructor(lin.weight), requires_grad=False)
        return lin

    return insert_subclass


def int8_dynamic_activation_int8_weight(device, layout_type=PlainLayoutType()):
    def apply_int8_dynamic_activation_int8_weight_quant(weight):
        in_features = weight.shape[1]
        # int8 dynamic quantization only has benefit when in_feature > 16
        if in_features <= 16:
            return weight.to(device)

        # weight settings
        mapping_type = MappingType.SYMMETRIC

        def get_weight_block_size(x):
            return (1, x.shape[1])

        target_dtype = torch.int8
        eps = torch.finfo(torch.float32).eps
        zero_point_dtype = torch.int64

        # input settings
        input_mapping_type = MappingType.SYMMETRIC
        input_target_dtype = torch.int8
        input_eps = 1e-5
        input_quant_min = -127
        input_quant_max = 127

        def input_quant_func(x):
            return to_affine_quantized(
                x,
                input_mapping_type,
                _get_per_token_block_size(x),
                input_target_dtype,
                eps=input_eps,
                quant_min=input_quant_min,
                quant_max=input_quant_max,
                scale_dtype=torch.float32 if x.dtype == torch.float16 else None,
            )

        block_size = get_weight_block_size(weight)
        weight = to_affine_quantized(
            weight.to("cuda"),
            mapping_type,
            block_size,
            target_dtype,
            eps=eps,
            zero_point_dtype=zero_point_dtype,
            layout_type=layout_type,
        )
        weight = to_linear_activation_quantized(weight, input_quant_func)
        return weight.to(device)

    return _get_linear_subclass_inserter(
        apply_int8_dynamic_activation_int8_weight_quant
    )


def int4_weight_only(device, group_size=64, inner_k_tiles=8):
    def apply_int4_weight_only_quant(weight: torch.Tensor):
        if weight.shape[-1] % group_size != 0:
            return weight

        mapping_type = MappingType.ASYMMETRIC
        block_size = (1, group_size)
        target_dtype = torch.int32
        quant_min = 0
        quant_max = 15
        eps = 1e-6
        preserve_zero = False
        zero_point_dtype = torch.bfloat16
        zero_point_domain = ZeroPointDomain.FLOAT
        layout_type = TensorCoreTiledLayoutType(inner_k_tiles=inner_k_tiles)
        return to_affine_quantized(
            weight.to("cuda"),
            mapping_type,
            block_size,
            target_dtype,
            quant_min,
            quant_max,
            eps,
            zero_point_dtype=zero_point_dtype,
            preserve_zero=preserve_zero,
            zero_point_domain=zero_point_domain,
            layout_type=layout_type,
        ).to(device)

    return _get_linear_subclass_inserter(apply_int4_weight_only_quant)


def quant_llm_fpx_weight_only(device, ebits: int, mbits: int):
    from torchao.prototype.quant_llm import QuantLlmLinearWeight

    def apply_quant_llm(weight: torch.Tensor) -> torch.Tensor:
        out_dim, in_dim = weight.shape
        if (in_dim % 64 != 0) or (out_dim % 256 != 0):
            return weight.to(device)
        return QuantLlmLinearWeight.from_float(weight.to("cuda"), ebits, mbits).to(
            device
        )

    return _get_linear_subclass_inserter(apply_quant_llm)


def int8_dynamic_activation_int4_weight(device, group_size=32):
    def apply_int8_dynamic_activation_int4_weight_quant(weight: torch.Tensor):
        if weight.shape[-1] % group_size != 0:
            return weight.to(device)

        # weight settings
        mapping_type = MappingType.SYMMETRIC
        block_size = (1, group_size)
        target_dtype = torch.int8
        eps = torch.finfo(torch.float32).eps
        quant_min = -8
        quant_max = 7

        # input settings
        input_mapping_type = MappingType.ASYMMETRIC
        input_target_dtype = torch.int8

        def input_quant_func(x):
            return to_affine_quantized(
                x, input_mapping_type, _get_per_token_block_size(x), input_target_dtype
            )

        weight = to_affine_quantized(
            weight.to("cuda"),
            mapping_type,
            block_size,
            target_dtype,
            quant_min,
            quant_max,
            eps,
        )
        weight = to_linear_activation_quantized(weight, input_quant_func)
        return weight.to(device)

    return _get_linear_subclass_inserter(
        apply_int8_dynamic_activation_int4_weight_quant
    )


def intx_weight_only(device, bit_size, group_size=64, pack_dim=-1):
    from torchao.prototype.uintx.Uintx import UintxLayoutType

    def apply_uintx_weight_only_quant(weight):
        layout_type = UintxLayoutType(bit_size=bit_size, pack_dim=pack_dim)
        mapping_type = MappingType.ASYMMETRIC
        block_size = (1, group_size)
        quant_min = 0
        quant_max = 2**bit_size - 1
        eps = torch.finfo(torch.float32).eps
        zero_point_dtype = torch.int32
        zero_point_domain = ZeroPointDomain.INT

        return to_affine_quantized(
            weight.to("cuda"),
            mapping_type,
            block_size,
            torch.uint8,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
            zero_point_dtype=zero_point_dtype,
            zero_point_domain=zero_point_domain,
            layout_type=layout_type,
        ).to(device)

    return _get_linear_subclass_inserter(apply_uintx_weight_only_quant)


def int8_weight_only(device):
    def apply_int8wo_quant(weight):
        mapping_type = MappingType.SYMMETRIC
        target_dtype = torch.int8
        eps = torch.finfo(torch.float32).eps
        zero_point_dtype = torch.int64
        block_size = (1, weight.shape[1])
        return to_affine_quantized(
            weight,
            mapping_type,
            block_size,
            target_dtype,
            eps=eps,
            zero_point_dtype=zero_point_dtype,
        ).to(device)

    return _get_linear_subclass_inserter(apply_int8wo_quant)


def float8_weight_only(device):
    def apply_fp8(weight: torch.Tensor):
        return Float8Tensor.from_float(weight).to(device)

    return _get_linear_subclass_inserter(apply_fp8)


class qdtype:
    def __init__(self, q, fn, defaults: list = []) -> None:
        self._q = q
        self.fn = fn
        self.special_variables = defaults

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, qdtype):
            return self.q == __value.q
        return False

    def __call__(self, *args) -> "qdtype":
        self.special_variables = list(args)
        return self

    @property
    def q(self) -> str:
        return self._q.replace("x", "x".join(map(str, self.special_variables)))

    def create(self, device: torch.device = "cuda") -> callable:
        if len(self.special_variables) != 0:
            return partial(self.fn, device, *self.special_variables)()
        return self.fn(device)


qfloatx = qdtype("floatx_", quant_llm_fpx_weight_only, [3, 2])
qfloat8 = qdtype("float8", float8_weight_only)
qintx = qdtype("intx_", intx_weight_only, [5])
qint4 = qdtype("int4", int4_weight_only)
qint4a8 = qdtype("int4a8", int8_dynamic_activation_int4_weight)
qint8 = qdtype("int8", int8_weight_only)
qint8a8 = qdtype("int8a8", int8_dynamic_activation_int8_weight, [PlainLayoutType()])
qint8a8ss = qdtype(
    "int8a8ss",
    int8_dynamic_activation_int8_weight,
    [SemiSparseLayoutType()],
)


def quantize_model(
    initialize: callable,
    dtype: qdtype,
    skip: list = [],
    strict_skip: bool = False,
    device: torch.device = "cuda",
    enabled: bool = True,
):
    model = initialize()
    if not enabled:
        return model

    def filter(module: torch.nn.Module, fqn: str) -> bool:
        return isinstance(module, torch.nn.Linear)

    if len(skip) != 0:
        if strict_skip:

            def filter(module: torch.nn.Module, fqn: str):
                return not any([s in fqn for s in skip]) and isinstance(
                    module, torch.nn.Linear
                )

        else:

            def filter(module: torch.nn.Module, fqn: str):
                return not any([fqn.startswith(s) for s in skip]) and isinstance(
                    module, torch.nn.Linear
                )

    quantize_(model, dtype.create(device), filter_fn=filter)
    return model
