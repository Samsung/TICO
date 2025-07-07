# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import torch._ops
    import torch.fx
import torch
import torch.fx.node

from tico.utils.utils import enforce_type

"""
This file includes OpArgs classes that provide arguments with type annotations.
- Each class provides type-checked arguments for the aten Op in the comment.
- Class name is determined by the follwoing priority.
    1. Torch spec (aten/src/ATen/native/native_functions.yaml in pytorch repo)
    2. pytorch doc (https://pytorch.org/docs/stable/index.html)
"""


@enforce_type
@dataclass
class AbsArgs:
    """
    abs(Tensor self) -> Tensor
    """

    input: torch.fx.Node


@enforce_type
@dataclass
class AddTensorArgs:
    """
    add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
    """

    input: Union[torch.fx.Node, float, int, torch.Tensor]
    other: Union[torch.fx.Node, float, int, torch.Tensor]


@enforce_type
@dataclass
class AddmmArgs:
    """
    addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
    """

    input: torch.fx.Node
    mat1: torch.fx.Node
    mat2: torch.fx.Node
    beta: Union[int, float] = 1
    alpha: Union[int, float] = 1


@enforce_type
@dataclass
class AliasCopyArgs:
    """
    alias_copy(Tensor self) -> Tensor
    """

    input: torch.fx.Node


@enforce_type
@dataclass
class AnyArgs:
    """
    any(Tensor self) -> Tensor
    any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
    any.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor
    """

    input: torch.fx.Node
    dim: Union[int, tuple, None] = None
    keepdim: bool = False


@enforce_type
@dataclass
class ArangeStartStepArgs:
    """
    arange.start_step(Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
    """

    start: Union[int, float]
    end: Union[int, float]
    step: Union[int, float] = 1


@enforce_type
@dataclass
class ArgMaxArgs:
    """
    argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
    """

    tensor: Union[torch.fx.Node, torch.Tensor]
    dim: Union[int, None] = None


@enforce_type
@dataclass
class AvgPool2dArgs:
    """
    avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> (Tensor)
    """

    input: torch.fx.Node
    kernel_size: List[int]
    stride: Optional[List[int]] = None
    padding: List[int] = field(default_factory=lambda: [0, 0])
    ceil_mode: bool = field(default=False)
    count_include_pad: bool = field(default=True)
    divisor_override: Optional[Union[int, None]] = None

    def __post_init__(self):
        assert len(self.kernel_size) == 2, len(self.kernel_size)
        self.stride = self.kernel_size if not self.stride else self.stride
        assert len(self.stride) == 2, len(self.stride)
        if self.padding is not None:
            assert len(self.padding) == 2, len(self.padding)
        if self.divisor_override is not None:
            assert isinstance(self.divisor_override, int), type(self.divisor_override)
            assert self.divisor_override != 0, f"Divisor must be not zero."


@enforce_type
@dataclass
class AdaptiveAvgPool2dArgs:
    """
    adaptive_avg_pool2d(Tensor self, SymInt[2] output_size) -> Tensor
    """

    input: torch.fx.Node
    output_size: List[int]


@enforce_type
@dataclass
class BmmArgs:
    """
    bmm(Tensor self, Tensor mat2) -> Tensor
    """

    input: torch.fx.Node
    mat2: torch.fx.Node


@enforce_type
@dataclass
class CatArgs:
    """
    cat(Tensor[] tensors, int dim=0) -> Tensor
    """

    tensors: List[torch.fx.Node]
    dim: int = 0


@enforce_type
@dataclass
class ClampArgs:
    """
    clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
    """

    input: torch.fx.Node
    min: Optional[Union[int, float]] = None
    max: Optional[Union[int, float]] = None


@enforce_type
@dataclass
class CloneArgs:
    """
    clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
    """

    input: torch.fx.Node
    memory_format: Optional[torch.memory_format] = None


@enforce_type
@dataclass
class ConstantPadNdArgs:
    """
    constant_pad_nd(Tensor self, SymInt[] pad, Scalar value=0) -> Tensor
    """

    input: torch.fx.Node
    pad: List[int]
    value: int | float


@enforce_type
@dataclass
class ConvArgs:
    """
    convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups) -> Tensor
    """

    input: torch.fx.Node
    weight: torch.fx.Node
    bias: Union[torch.fx.Node, None]
    stride: List[int]
    padding: List[int]
    dilation: List[int]
    transposed: bool
    output_padding: List[int]
    groups: int


@enforce_type
@dataclass
class ConvTranspose2DArgs:
    """
    conv_transpose2d.input(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] output_padding=0, SymInt groups=1, SymInt[2] dilation=1) -> Tensor
    """

    input: torch.fx.Node
    weight: torch.fx.Node
    bias: Union[torch.fx.Node, None] = None
    stride: List[int] = field(default_factory=lambda: [1, 1])
    padding: List[int] = field(default_factory=lambda: [0, 0])
    output_padding: List[int] = field(default_factory=lambda: [0, 0])
    groups: int = 1
    dilation: List[int] = field(default_factory=lambda: [1, 1])

    def __post_init__(self):
        assert len(self.stride) == 2, len(self.stride)
        assert len(self.dilation) == 2, len(self.dilation)


@enforce_type
@dataclass
class Conv2DArgs:
    """
    conv2d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] dilation=1, SymInt groups=1) -> Tensor
    conv2d.padding(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, str padding="valid", SymInt[2] dilation=1, SymInt groups=1) -> Tensor
    """

    input: torch.fx.Node
    weight: torch.fx.Node
    bias: Union[torch.fx.Node, None] = None
    stride: List[int] = field(default_factory=lambda: [1, 1])
    padding: Union[List[int], str] = field(default_factory=lambda: [0, 0])
    dilation: List[int] = field(default_factory=lambda: [1, 1])
    groups: int = 1

    def __post_init__(self):
        assert len(self.stride) == 2, len(self.stride)
        assert len(self.dilation) == 2, len(self.dilation)


@enforce_type
@dataclass
class Conv1DArgs:
    """
    conv1d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[1] stride=1, SymInt[1] padding=0, SymInt[1] dilation=1, SymInt groups=1) -> Tensor
    conv1d.padding(Tensor input, Tensor weight, Tensor? bias=None, SymInt[1] stride=1, str padding="valid", SymInt[1] dilation=1, SymInt groups=1) -> Tensor
    """

    input: torch.fx.Node
    weight: torch.fx.Node
    bias: Union[torch.fx.Node, None] = None
    stride: List[int] = field(default_factory=lambda: [1])
    padding: Union[List[int], str] = field(default_factory=lambda: [0])
    dilation: List[int] = field(default_factory=lambda: [1])
    groups: int = 1

    def __post_init__(self):
        assert len(self.stride) == 1, len(self.stride)
        assert len(self.dilation) == 1, len(self.dilation)


@enforce_type
@dataclass
class CopyArgs:
    """
    copy(Tensor self, Tensor src, bool non_blocking=False) -> Tensor
    """

    dst: torch.fx.Node
    src: torch.fx.Node


@enforce_type
@dataclass
class CosArgs:
    """
    cos(Tensor self) -> Tensor
    """

    input: torch.fx.Node


@enforce_type
@dataclass
class CumsumArgs:
    """
    cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
    """

    input: torch.fx.Node
    dim: int


@enforce_type
@dataclass
class DequantizePerChannelArgs:
    """
    quantized_decomposed.dequantize_per_channel(Tensor input, Tensor scales, Tensor? zero_points, int axis, int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor
    """

    input: torch.fx.Node
    scales: torch.fx.Node
    zero_points: torch.fx.Node
    axis: int
    quant_min: int
    quant_max: int
    dtype: torch.dtype


@enforce_type
@dataclass
class DequantizePerTensorArgs:
    """
    quantized_decomposed.dequantize_per_tensor(input: TensorBox, scale: float, zero_point: int, quant_min: int, quant_max: int, dtype: torch.dtype) -> TensorBox
    """

    input: torch.fx.Node
    scale: float
    zero_point: int
    quant_min: int
    quant_max: int
    dtype: torch.dtype


@enforce_type
@dataclass
class DivTensorArgs:
    """
    div.Tensor(Tensor self, Tensor other) -> Tensor
    """

    input: Union[torch.fx.Node, float, int, torch.Tensor]
    other: Union[torch.fx.Node, float, int, torch.Tensor]


@enforce_type
@dataclass
class EmbeddingArgs:
    """
    embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor
    """

    weight: torch.fx.Node
    indices: torch.fx.Node
    padding_idx: int = 1
    scale_grad_by_freq: bool = False
    sparse: bool = False


@enforce_type
@dataclass
class EqArgs:
    """
    eq.Scalar(Tensor self, Scalar other) -> Tensor
    eq.Tensor(Tensor self, Tensor other) -> Tensor
    """

    input: Union[torch.fx.Node, torch.Tensor, float, int]
    other: Union[torch.fx.Node, torch.Tensor, float, int]


@enforce_type
@dataclass
class ExpArgs:
    """
    exp(Tensor self) -> Tensor
    """

    input: torch.fx.Node


@enforce_type
@dataclass
class ExpandArgs:
    """
    expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)
    expand_copy(Tensor self, SymInt[] size, *, bool implicit=False) -> Tensor
    """

    input: torch.fx.Node
    size: List[int]


@enforce_type
@dataclass
class FakeQuantizePerChannelArgs:
    """
    fake_quantize_per_channel_affine(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max) -> Tensor
    fake_quantize_per_channel_affine_cachemask(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max) -> (Tensor output, Tensor mask)
    """

    input: torch.fx.Node
    scale: torch.fx.Node
    zero_point: torch.fx.Node
    axis: int
    quant_min: int
    quant_max: int


@enforce_type
@dataclass
class FakeQuantizePerTensorTQParamArgs:
    """
    fake_quantize_per_tensor_affine.tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, int quant_min, int quant_max) -> Tensor
    """

    input: torch.fx.Node
    scale: torch.fx.Node
    zero_point: torch.fx.Node
    quant_min: int
    quant_max: int


@enforce_type
@dataclass
class FullLikeArgs:
    """
    full_like(Tensor self, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
    """

    input: torch.fx.Node
    fill_value: Union[int, float, bool]
    pin_memory: Optional[bool] = None
    memory_format: Optional[torch.memory_format] = None


@enforce_type
@dataclass
class FullArgs:
    """
    full(SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
    """

    size: Union[list, tuple, torch.Size]
    fill_value: Union[int, float, bool]


@enforce_type
@dataclass
class GeArgs:
    """
    ge.Scalar(Tensor self, Scalar other) -> Tensor
    ge.Tensor(Tensor self, Tensor other) -> Tensor
    """

    input: Union[torch.fx.Node, torch.Tensor, float, int]
    other: Union[torch.fx.Node, torch.Tensor, float, int]


@enforce_type
@dataclass
class GeluArgs:
    """
    gelu(Tensor self, *, str approximate='none') -> Tensor
    """

    input: torch.fx.Node
    approximate: Optional[str] = "none"


@enforce_type
@dataclass
class GtArgs:
    """
    gt.Scalar(Tensor self, Scalar other) -> Tensor
    gt.Tensor(Tensor self, Tensor other) -> Tensor
    """

    input: Union[torch.fx.Node, torch.Tensor, float, int]
    other: Union[torch.fx.Node, torch.Tensor, float, int]


@enforce_type
@dataclass
class HardTanhArgs:
    """
    hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor
    """

    input: Union[torch.fx.Node, torch.Tensor]
    min_val: Union[float, int] = -1
    max_val: Union[float, int] = 1


@enforce_type
@dataclass
class IndexSelectArgs:
    """
    index_select(Tensor self, int dim, Tensor index) -> Tensor
    """

    input: torch.fx.Node
    dim: int
    index: Union[torch.fx.Node, torch.Tensor]


@enforce_type
@dataclass
class IndexArgs:
    """
    index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
    """

    input: torch.fx.Node
    indices: List[Union[torch.fx.Node, torch.Tensor, int, None]]


@enforce_type
@dataclass
class InstanceNormArgs:
    """
    instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor
    """

    input: torch.fx.Node
    weight: Optional[torch.fx.Node]
    bias: Optional[torch.fx.Node]
    running_mean: Optional[torch.fx.Node]
    running_var: Optional[torch.fx.Node]
    use_input_stats: bool
    momentum: float
    eps: float
    cudnn_enabled: bool


@enforce_type
@dataclass
class LeakyReluArgs:
    """
    leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor
    """

    input: torch.fx.Node
    negative_slope: float = 0.01


@enforce_type
@dataclass
class LinearArgs:
    """
    linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
    """

    input: torch.fx.Node
    weight: torch.fx.Node
    bias: Optional[torch.fx.Node] = None


@enforce_type
@dataclass
class LogArgs:
    """
    log(Tensor self) -> Tensor
    """

    input: torch.fx.Node


@enforce_type
@dataclass
class Log1pArgs:
    """
    log1p(Tensor self) -> Tensor
    """

    input: torch.fx.Node


@enforce_type
@dataclass
class LogicalAndArgs:
    """
    logical_and(Tensor self, Tensor other) -> Tensor
    """

    input: torch.fx.Node
    other: torch.fx.Node


@enforce_type
@dataclass
class LogicalNotArgs:
    """
    logical_not(Tensor self) -> Tensor
    """

    input: torch.fx.Node


@enforce_type
@dataclass
class LtArgs:
    """
    lt.Tensor(Tensor self, Tensor other) -> Tensor
    """

    input: torch.fx.Node
    other: torch.fx.Node


@enforce_type
@dataclass
class MaxDimArgs:
    """
    max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
    """

    input: torch.fx.Node
    dim: int
    keepdim: bool = False


@enforce_type
@dataclass
class MaxPool2dWithIndicesArgs:
    """
    max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
    """

    input: torch.fx.Node
    kernel_size: List[int]
    stride: Optional[List[int]] = None
    padding: List[int] = field(default_factory=lambda: [0, 0])
    dilation: List[int] = field(default_factory=lambda: [1, 1])
    ceil_mode: bool = field(default=False)

    def __post_init__(self):
        assert len(self.kernel_size) == 2, len(self.kernel_size)
        self.stride = self.kernel_size if not self.stride else self.stride
        assert len(self.stride) == 2, len(self.stride)
        if self.padding is not None:
            assert len(self.padding) == 2, len(self.padding)
        if self.dilation is not None:
            assert len(self.dilation) == 2, len(self.dilation)


@enforce_type
@dataclass
class MeanDimArgs:
    """
    mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    """

    input: torch.fx.Node
    dim: List[int]
    keepdim: bool = False
    dtype: Optional[torch.dtype] = None


@enforce_type
@dataclass
class MatmulArgs:
    """
    mm(Tensor self, Tensor mat2) -> Tensor
    """

    input: torch.fx.Node
    other: torch.fx.Node


@enforce_type
@dataclass
class MaximumArgs:
    """
    maximum(Tensor self, Tensor other) -> Tensor
    """

    input: Union[torch.fx.Node, torch.Tensor]
    other: Union[torch.fx.Node, torch.Tensor]


@enforce_type
@dataclass
class MinimumArgs:
    """
    minimum(Tensor self, Tensor other) -> Tensor
    """

    input: Union[torch.fx.Node, torch.Tensor]
    other: Union[torch.fx.Node, torch.Tensor]


@enforce_type
@dataclass
class MulTensorArgs:
    """
    mul.Tensor(Tensor self, Tensor other) -> Tensor
    """

    input: Union[torch.fx.Node, torch.Tensor, int, float]
    other: Union[torch.fx.Node, torch.Tensor, int, float]


@enforce_type
@dataclass
class MulScalarArgs:
    """
    mul.Scalar(Tensor self, Scalar other) -> Tensor
    """

    input: torch.fx.Node
    other: Union[int, float]


@enforce_type
@dataclass
class NeScalarArgs:
    """
    ne.Scalar(Tensor self, Scalar other) -> Tensor
    """

    input: Union[torch.fx.Node, torch.Tensor, float, int, bool]
    other: Union[torch.fx.Node, torch.Tensor, float, int, bool]


@enforce_type
@dataclass
class NativeBatchNormLegitNoTrainingArgs:
    """
    _native_batch_norm_legit_no_training    (Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor)
    """

    input: torch.fx.Node
    weight: Optional[torch.fx.Node]
    bias: Optional[torch.fx.Node]
    running_mean: Optional[torch.fx.Node]
    running_var: Optional[torch.fx.Node]
    momentum: float
    eps: float


@enforce_type
@dataclass
class NativeGroupNormArgs:
    """
    native_group_norm(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps) -> (Tensor, Tensor, Tensor)
    """

    input: torch.fx.Node
    weight: Optional[torch.fx.Node]
    bias: Optional[torch.fx.Node]
    N: int
    C: int
    HxW: int
    group: int
    eps: float


@enforce_type
@dataclass
class NativeLayerNormArgs:
    """
    native_layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)
    """

    input: torch.fx.Node
    normalized_shape: Union[tuple, list]
    weight: Optional[torch.fx.Node]
    bias: Optional[torch.fx.Node]
    eps: float


@enforce_type
@dataclass
class NeTensorArgs:
    """
    ne.Tensor(Tensor self, Tensor other) -> Tensor
    """

    input: Union[torch.fx.Node, torch.Tensor, float, int, bool]
    other: Union[torch.fx.Node, torch.Tensor, float, int, bool]


@enforce_type
@dataclass
class NegArgs:
    """
    neg(Tensor self) -> Tensor
    """

    input: torch.fx.Node


@enforce_type
@dataclass
class PermuteArgs:
    """
    permute(Tensor(a) self, int[] dims) -> Tensor(a)
    """

    input: torch.fx.Node
    dims: List[int]


@enforce_type
@dataclass
class PowTensorTensorArgs:
    """
    pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
    """

    input: torch.fx.Node
    exponent: Union[torch.fx.Node]


@enforce_type
@dataclass
class PowTensorScalarArgs:
    """
    pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
    """

    input: torch.fx.Node
    exponent: Union[float, int]


@enforce_type
@dataclass
class PReLUArgs:
    """
    prelu(Tensor self, Tensor weight) -> Tensor
    """

    input: torch.fx.Node
    weight: torch.fx.Node


@enforce_type
@dataclass
class QuantizePerTensorArgs:
    """
    quantized_decomposed.quantize_per_tensor(input: TensorBox, scale: float, zero_point: int, quant_min: int, quant_max: int, dtype: torch.dtype) -> TensorBox
    """

    tensor: torch.fx.Node
    scale: float
    zero_p: int
    quant_min: int
    quant_max: int
    dtype: torch.dtype


@enforce_type
@dataclass
class ReciprocalArgs:
    """
    reciprocal(Tensor self) -> Tensor
    """

    input: torch.fx.Node


@enforce_type
@dataclass
class ReluArgs:
    """
    relu(Tensor self) -> Tensor
    """

    input: torch.fx.Node


@enforce_type
@dataclass
class Relu6Args:
    """
    relu6(Tensor self) -> Tensor
    """

    input: torch.fx.Node


@enforce_type
@dataclass
class RepeatArgs:
    """
    repeat(Tensor self, SymInt[] repeats) -> Tensor
    """

    input: torch.fx.Node
    repeats: List[int]


@enforce_type
@dataclass
class ReshapeArgs:
    """
    reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)
    """

    input: torch.fx.Node
    shape: List[int]


@enforce_type
@dataclass
class ResizeNearestNeighborArgs:
    """
    # Mapped from `torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode='nearest')` case.
    """

    input: torch.fx.Node
    size: List[int]


@enforce_type
@dataclass
class RoundArgs:
    """
    round(Tensor self) -> Tensor
    """

    input: torch.fx.Node


@enforce_type
@dataclass
class RsqrtArgs:
    """
    rsqrt(Tensor self) -> Tensor
    """

    input: torch.fx.Node


@enforce_type
@dataclass
class ScalarTensorArgs:
    """
    scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
    """

    scalar: Union[int, float]


@enforce_type
@dataclass
class SelectCopyIntArgs:
    """
    select_copy.int(Tensor self, int dim, SymInt index) -> Tensor
    """

    input: torch.fx.Node
    dim: int
    index: int


@enforce_type
@dataclass
class SigmoidArgs:
    """
    sigmoid(Tensor self) -> Tensor
    """

    input: torch.fx.Node


@enforce_type
@dataclass
class SinArgs:
    """
    sin(Tensor self) -> Tensor
    """

    input: torch.fx.Node


@enforce_type
@dataclass
class SliceArgs:
    """
    slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)
    slice_copy.Tensor(Tensor self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor
    """

    input: torch.fx.Node
    dim: int = 0
    start: Optional[int] = None
    end: Optional[int] = None
    step: Optional[int] = 1


@enforce_type
@dataclass
class SafeSoftmaxArgs:
    """
    _safe_softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
    """

    input: torch.fx.Node
    dim: int
    dtype: Optional[torch.dtype] = None


@enforce_type
@dataclass
class SoftmaxArgs:
    """
    _softmax(Tensor self, int dim, bool half_to_float) -> Tensor
    """

    input: torch.fx.Node
    dim: int
    half_to_float: bool


@enforce_type
@dataclass
class SplitWithSizesArgs:
    """
    split_with_sizes(Tensor(a->*) self, SymInt[] split_sizes, int dim=0) -> Tensor(a)[]
    """

    input: torch.fx.Node
    split_sizes: List[int]
    dim: int = 0


@enforce_type
@dataclass
class SqrtArgs:
    """
    sqrt(Tensor self) -> Tensor
    """

    input: torch.fx.Node


@enforce_type
@dataclass
class SqueezeArgs:
    """
    squeeze.dims(Tensor(a) self, int[] dim) -> Tensor(a)
    squeeze_copy.dims(Tensor self, int[] dim) -> Tensor
    """

    input: torch.fx.Node
    dims: List[int] = field(default_factory=lambda: [])


@enforce_type
@dataclass
class SubTensorArgs:
    """
    sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
    """

    input: Union[torch.fx.Node, torch.Tensor, float, int]
    other: Union[torch.fx.Node, torch.Tensor, float, int]
    alpha: Optional[int] = None


@enforce_type
@dataclass
class SumDimIntListArgs:
    """
    sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    """

    input: Union[torch.fx.Node, torch.Tensor, float, int]
    dim: List[int] = field(default_factory=list)
    keepdim: bool = False
    dtype: Optional[torch.dtype] = None


@enforce_type
@dataclass
class TanhArgs:
    """
    tanh(Tensor self) -> Tensor
    """

    input: torch.fx.Node


@enforce_type
@dataclass
class ToCopyArgs:
    """
    _to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor
    """

    input: torch.fx.Node
    dtype: Optional[torch.dtype] = None
    layout: Optional[torch.layout] = None
    device: Optional[torch.device] = None
    pin_memory: Optional[bool] = None
    non_blocking: Optional[bool] = False
    memory_format: Optional[torch.memory_format] = None


@enforce_type
@dataclass
class ToDtypeArgs:
    """
    to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
    """

    input: torch.fx.Node
    dtype: Optional[torch.dtype] = None
    non_blocking: Optional[bool] = False
    copy: Optional[bool] = False
    memory_format: Optional[torch.memory_format] = None


@enforce_type
@dataclass
class ToDtypeLayoutArgs:
    """
    to.dtype_layout(Tensor(a) self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
    """

    input: torch.fx.Node
    dtype: Optional[torch.dtype] = None
    layout: Optional[torch.layout] = None
    device: Optional[torch.device] = None
    pin_memory: Optional[bool] = None
    non_blocking: Optional[bool] = False
    copy: Optional[bool] = False
    memory_format: Optional[torch.memory_format] = None


@enforce_type
@dataclass
class UnSqueezeArgs:
    """
    unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
    unsqueeze_copy(Tensor self, int dim) -> Tensor
    """

    input: torch.fx.Node
    dim: int


@enforce_type
@dataclass
class UpsampleNearest2DVecArgs:
    """
    upsample_nearest2d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor
    """

    input: torch.fx.Node
    output_size: Optional[List[int]]
    scale_factors: Optional[List[float]]


@enforce_type
@dataclass
class ViewArgs:
    """
    view(Tensor(a) self, SymInt[] size) -> Tensor(a)
    view_copy(Tensor self, SymInt[] size) -> Tensor
    """

    input: torch.fx.Node
    size: List[int]


@enforce_type
@dataclass
class WhereSelfArgs:
    """
    where.self(Tensor condition, Tensor self, Tensor other) -> Tensor
    """

    condition: torch.fx.Node
    input: Union[torch.fx.Node, torch.Tensor]
    other: Union[torch.fx.Node, torch.Tensor]
