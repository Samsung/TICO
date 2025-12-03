import torch

batch = 1
groups = 14
out_channels = 70
in_channels = 28
kernel = 6
padding = 5  # 0 <= padding < kernel
dilation = 3
stride = 5

bias = torch.rand(out_channels)
inp = torch.randn(batch, in_channels, 25)
w = torch.randn(in_channels, out_channels // groups, kernel)

e_out = torch.nn.functional.conv_transpose1d(
    inp, w, bias=bias, groups=groups, stride=stride, padding=padding, dilation=dilation
)

if groups > 1:
    w_conv_transposed = torch.zeros(out_channels, in_channels // groups, kernel)
    for i in range(0, groups):
        w_conv_transposed[
            i * out_channels // groups : (i + 1) * out_channels // groups, :, :
        ] = (
            w[i * in_channels // groups : (i + 1) * in_channels // groups, :, :]
            .transpose(1, 0)
            .flip(-1)
        )
else:
    w_conv_transposed = w.transpose(1, 0).flip(-1)

inp_strided = torch.zeros(
    inp.shape[0],
    inp.shape[1],
    stride * (inp.shape[2] - 1) - 2 * padding + 2 * dilation * (kernel - 1) + 1,
)
indices = torch.arange(0, inp.shape[2])
inp_strided[:, :, stride * indices + dilation * (kernel - 1) - padding] = inp[
    :, :, indices
]
e_out_c = torch.nn.functional.conv1d(
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
