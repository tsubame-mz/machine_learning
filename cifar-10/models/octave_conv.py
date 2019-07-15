import torch.nn as nn


class OctaveConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        alpha_in=0.5,
        alpha_out=0.5,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(OctaveConv, self).__init__()
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        assert 0 <= alpha_in < 1 and 0 <= alpha_out < 1, "Alphas should be in the interval from 0 to 1."

        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.stride = stride

        l_in_channels = int(alpha_in * in_channels)
        h_in_channels = in_channels - l_in_channels
        l_out_channels = int(alpha_out * out_channels)
        h_out_channels = out_channels - l_out_channels
        assert h_in_channels > 0 and h_out_channels > 0, "High channels should be more than 0"

        self.conv_h2h = nn.Conv2d(h_in_channels, h_out_channels, kernel_size, 1, padding, dilation, groups, bias)
        self.conv_h2l = (
            None
            if (l_out_channels == 0)
            else nn.Conv2d(h_in_channels, l_out_channels, kernel_size, 1, padding, dilation, groups, bias)
        )
        self.conv_l2h = (
            None
            if (l_in_channels == 0)
            else nn.Conv2d(l_in_channels, h_out_channels, kernel_size, 1, padding, dilation, groups, bias)
        )
        self.conv_l2l = (
            None
            if (l_in_channels == 0) or (l_out_channels == 0)
            else nn.Conv2d(l_in_channels, l_out_channels, kernel_size, 1, padding, dilation, groups, bias)
        )

        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        x_h, x_l = x if type(x) is tuple else (x, None)
        # High
        if x_h is not None:
            x_h = self.downsample(x_h) if self.stride == 2 else x_h

            # High -> High
            x_h2h = self.conv_h2h(x_h)
            # print("x_h2h:", x_h2h.shape)

            # high -> Low
            if self.conv_h2l is not None:
                x_h2l = self.conv_h2l(self.downsample(x_h))
                # print("x_h2l:", x_h2l.shape)
            else:
                x_h2l = None
        # Low
        if x_l is not None:
            # Low -> High
            if self.conv_l2h is not None:
                x_l2h = self.conv_l2h(x_l)
                x_l2h = self.upsample(x_l2h) if self.stride == 1 else x_l2h
                # print("x_l2h:", x_l2h.shape)
            else:
                # assert?
                x_l2h = None

            # Low -> Low
            if self.conv_l2l is not None:
                x_l2l = self.downsample(x_l) if self.stride == 2 else x_l
                x_l2l = self.conv_l2l(x_l2l)
                # print("x_l2l:", x_l2l.shape)
            else:
                x_l2l = None

            x_h = x_h2h + x_l2h
            x_l = x_h2l + x_l2l if x_h2l is not None and x_l2l is not None else None
            return x_h, x_l
        else:
            return x_h2h, x_h2l


if __name__ == "__main__":
    import torch

    octconv = OctaveConv(3, 8, 3, alpha_in=0, alpha_out=0.25, stride=1, padding=1)
    print(octconv)
    input_tsr = torch.randn(1, 3, 32, 32)
    print("input_tsr:", input_tsr.shape)
    output_tsr_h, output_tsr_l = octconv(input_tsr)
    print("output_tsr_h:", output_tsr_h.shape)
    print("output_tsr_l:", output_tsr_l.shape)

    print("-" * 80)
    octconv_2 = OctaveConv(8, 8, 3, alpha_in=0.25, alpha_out=0.25, stride=1, padding=1)
    print(octconv_2)
    output_tsr_h, output_tsr_l = octconv_2((output_tsr_h, output_tsr_l))
    print("output_tsr_h:", output_tsr_h.shape)
    print("output_tsr_l:", output_tsr_l.shape)

    print("-" * 80)
    octconv_2_1 = OctaveConv(8, 8, 3, alpha_in=0.25, alpha_out=0.25, stride=1, padding=1)
    print(octconv_2_1)
    output_tsr_h, output_tsr_l = octconv_2_1((output_tsr_h, output_tsr_l))
    print("output_tsr_h:", output_tsr_h.shape)
    print("output_tsr_l:", output_tsr_l.shape)

    print("-" * 80)
    octconv_3 = OctaveConv(8, 8, 3, alpha_in=0.25, alpha_out=0, stride=1, padding=1)
    print(octconv_3)
    output_tsr_h, _ = octconv_3((output_tsr_h, output_tsr_l))
    print("output_tsr_h:", output_tsr_h.shape)
