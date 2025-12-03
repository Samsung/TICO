import torch

batch = 1
groups = 2
out_channels = 128
in_channels = 256
kernel = (3, 8)
padding = (1, 2)  # 0 <= padding < kernel
dilation = (3, 2)
stride = (2, 3)

bias = torch.rand(out_channels)
inp = torch.randn(batch, in_channels, 21, 80)
w = torch.randn(in_channels, out_channels // groups, kernel[0], kernel[1])

e_out = torch.nn.functional.conv_transpose2d(
    inp, w, bias=bias, groups=groups, stride=stride, padding=padding, dilation=dilation
)

if groups > 1:
    w_conv_transposed = torch.zeros(
        out_channels, in_channels // groups, kernel[0], kernel[1]
    )
    for i in range(0, groups):
        w_conv_transposed[
            i * out_channels // groups : (i + 1) * out_channels // groups, :, :, :
        ] = (
            w[i * in_channels // groups : (i + 1) * in_channels // groups, :, :, :]
            .transpose(1, 0)
            .flip((-2, -1))
        )
    w_orig = torch.zeros_like(w)
    for i in range(0, groups):
        w_orig[i * in_channels // groups : (i + 1) * in_channels // groups, :, :, :] = (
            w_conv_transposed[
                i * out_channels // groups : (i + 1) * out_channels // groups, :, :, :
            ]
            .transpose(1, 0)
            .flip((-2, -1))
        )
    error_tr = torch.max(torch.abs(w - w_orig))

else:
    w_conv_transposed = w.transpose(1, 0).flip((-2, -1))
    w_orig = w_conv_transposed.transpose(1, 0).flip((-2, -1))
    error_tr = torch.max(torch.abs(w - w_orig))

strided_pad = (
    dilation[0] * (kernel[0] - 1) - padding[0],
    dilation[1] * (kernel[1] - 1) - padding[1],
)

inp_strided = torch.zeros(
    inp.shape[0],
    inp.shape[1],
    stride[0] * (inp.shape[2] - 1) + 2 * strided_pad[0] + 1,
    stride[1] * (inp.shape[3] - 1) + 2 * strided_pad[1] + 1,
)

# input will ne interleaved with zero rows and columns
indices = torch.arange(0, inp.shape[2])
inp_strided[
    :,
    :,
    stride[0] * indices + strided_pad[0],
    strided_pad[1] : -strided_pad[1] : stride[1],
] = inp[:, :, indices, :]

e_out_c = torch.nn.functional.conv2d(
    inp_strided,
    w_conv_transposed,
    bias=bias,
    groups=groups,
    dilation=dilation,
    padding=0,
)

error_max = torch.max(torch.abs(e_out - e_out_c))
error_mean = torch.mean(torch.abs(e_out - e_out_c))
print(
    f"error_max - {error_max.cpu().item():.4f}, error_mean - {error_mean.cpu().item():.4f}"
)
