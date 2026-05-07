import torch
import torch.nn as nn


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize

    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CA(nn.Module):
    def __init__(self, inp, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        _n, _c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class DSConv_CA(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1):
        super().__init__()
        self.has_shortcut = s == 1 and c1 == c2
        expanded_c = c1 * expansion
        self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
        self.dw_conv = Conv(expanded_c, expanded_c, k=k, s=s, g=expanded_c)

        self.ca = CA(expanded_c, expanded_c)

        self.cv1 = Conv(expanded_c, c2, k=1, s=1)

    def forward(self, x):

        y = self.expansion_conv(x)
        y = self.dw_conv(y)
        y = self.ca(y)
        y = self.cv1(y)
        if self.has_shortcut:
            y += x

        return y


class CA_DSConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1):
        super().__init__()
        self.has_shortcut = s == 1 and c1 == c2
        expanded_c = c1 * expansion
        self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
        self.ca = CA(expanded_c, expanded_c)
        self.dw_conv = Conv(expanded_c, expanded_c, k=k, s=s, g=expanded_c)
        self.cv1 = Conv(expanded_c, c2, k=1, s=1)

    def forward(self, x):

        y = self.expansion_conv(x)
        y = self.ca(y)
        y = self.dw_conv(y)
        y = self.cv1(y)
        if self.has_shortcut:
            y += x

        return y


class SPDConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.d = d
        self.out = c2
        self.conv = Conv(c1 * 4, c2, k, s, p=p, g=g, d=d, act=act)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

        return self.conv(x)


class SPD(nn.Module):
    def __init__(self, c1, c2=None, d=1):
        super().__init__()
        self.d = d
        # 计算输出通道数：输入通道数乘以4（因为空间拆分成4份）
        self.c1 = c1
        self.c2 = c1 * 4  # 输出通道数

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


#         size_tensor = x.size()
#         return torch.cat([x[...,0:size_tensor[2]//2,0:size_tensor[3]//2],
#                          x[...,0:size_tensor[2]//2,size_tensor[3]//2:],
#                          x[...,size_tensor[2]//2:,0:size_tensor[3]//2],
#                          x[...,size_tensor[2]//2:,size_tensor[3]//2:]  ],1)


class SPDConv_CA(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1):
        super().__init__()
        self.has_shortcut = s == 1 and c1 == c2
        expanded_c = c1 * expansion
        self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
        self.spd_conv = SPDConv(expanded_c, expanded_c, k=1, s=1)

        self.ca = CA(expanded_c, expanded_c)
        self.cv1 = Conv(expanded_c, c2, k=1, s=1)

    def forward(self, x):

        y = self.expansion_conv(x)
        y = self.spd_conv(y)
        y = self.ca(y)
        y = self.cv1(y)

        if self.has_shortcut:
            y += x

        return y


class CA_SPDConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1):
        super().__init__()
        self.has_shortcut = s == 1 and c1 == c2
        expanded_c = c1 * expansion
        self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
        self.ca = CA(expanded_c, expanded_c)
        self.spd_conv = SPDConv(expanded_c, c2, k=1, s=1)
        self.cv1 = Conv(expanded_c, c2, k=1, s=1)

    def forward(self, x):

        y = self.expansion_conv(x)
        y = self.ca(y)
        y = self.spd_conv(y)

        if self.has_shortcut:
            y += x

        return y


class DWTForward(nn.Module):
    def __init__(self, J=1, wave="haar", mode="zero"):
        super().__init__()
        self.J = J
        # 直接定义4维卷积核，自动跟随模型精度/设备
        self.register_buffer("kernel_ll", torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]], dtype=torch.float32) / 2.0)
        self.register_buffer("kernel_lh", torch.tensor([[[[1.0, -1.0], [1.0, -1.0]]]], dtype=torch.float32) / 2.0)
        self.register_buffer("kernel_hl", torch.tensor([[[[1.0, 1.0], [-1.0, -1.0]]]], dtype=torch.float32) / 2.0)
        self.register_buffer("kernel_hh", torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]], dtype=torch.float32) / 2.0)

    def forward(self, x):
        B, C, H, W = x.shape
        # 确保尺寸是偶数
        assert H % 2 == 0 and W % 2 == 0, "DWT: 高宽必须是2的倍数"

        #  reshape 成每组单通道卷积
        x_r = x.reshape(B * C, 1, H, W)

        # 小波卷积（自动匹配 x 的 dtype 和 device）
        yl = nn.functional.conv2d(x_r, self.kernel_ll, stride=2)
        yh1 = nn.functional.conv2d(x_r, self.kernel_lh, stride=2)
        yh2 = nn.functional.conv2d(x_r, self.kernel_hl, stride=2)
        yh3 = nn.functional.conv2d(x_r, self.kernel_hh, stride=2)

        # 恢复形状
        yl = yl.reshape(B, C, H // 2, W // 2)
        yh1 = yh1.reshape(B, C, H // 2, W // 2)
        yh2 = yh2.reshape(B, C, H // 2, W // 2)
        yh3 = yh3.reshape(B, C, H // 2, W // 2)

        # 输出格式对齐 pytorch_wavelets (yl, (yh,))
        yh = torch.stack([yh1, yh2, yh3], dim=2)
        return yl, (yh,)


class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.wt = DWTForward(J=1, mode="zero", wave="haar")
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(
                in_ch * 4, out_ch, kernel_size=1, stride=1, bias=False
            ),  # 加bias=False，和BN层搭配无意义，节省显存+加速
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)

        y_HL = yH[0][:, :, 0, ...]
        y_LH = yH[0][:, :, 1, ...]
        y_HH = yH[0][:, :, 2, ...]
        # 拼接低频+3个高频分量 → 通道数变成 in_ch*4
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x


class HWD_CA(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1):
        super().__init__()
        self.has_shortcut = s == 1 and c1 == c2
        expanded_c = c1 * expansion
        self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
        self.hwd = Down_wt(expanded_c, expanded_c)

        self.ca = CA(expanded_c, expanded_c)

        self.cv1 = Conv(expanded_c, c2, k=1, s=1)

    def forward(self, x):

        y = self.expansion_conv(x)
        y = self.hwd(y)
        y = self.ca(y)
        y = self.cv1(y)
        if self.has_shortcut:
            y += x

        return y


class CA_HWD(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1, drop_connect_rate=0.1):
        super().__init__()
        self.has_shortcut = s == 1 and c1 == c2
        expanded_c = c1 * expansion
        self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
        self.ca = CA(expanded_c, expanded_c)
        self.hwd = Down_wt(expanded_c, c2)
        # self.cv1 = Conv(expanded_c, c2, k=1, s=1)
        self.drop_connect_rate = drop_connect_rate
        if self.has_shortcut and drop_connect_rate > 0:
            self.dropout = DropPath(drop_connect_rate)

    def forward(self, x):

        y = self.expansion_conv(x)
        y = self.ca(y)
        y = self.hwd(y)

        if self.has_shortcut:
            if self.drop_connect_rate > 0:
                y = self.dropout(y)
            y += x

        return y


class Conv_CA(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1):
        super().__init__()
        self.has_shortcut = s == 1 and c1 == c2
        expanded_c = c1 * expansion
        self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
        self.conv = Conv(expanded_c, expanded_c, k=k, s=s)

        self.ca = CA(expanded_c, expanded_c)

        self.cv1 = Conv(expanded_c, c2, k=1, s=1)

    def forward(self, x):

        y = self.expansion_conv(x)
        y = self.conv(y)
        y = self.ca(y)
        y = self.cv1(y)
        if self.has_shortcut:
            y += x

        return y


class CA_Conv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1):
        super().__init__()
        self.has_shortcut = s == 1 and c1 == c2
        expanded_c = c1 * expansion
        self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
        self.ca = CA(expanded_c, expanded_c)
        self.conv = Conv(expanded_c, c2, k=k, s=s)
        self.cv1 = Conv(expanded_c, c2, k=1, s=1)

    def forward(self, x):

        y = self.expansion_conv(x)
        y = self.ca(y)
        y = self.conv(y)

        if self.has_shortcut:
            y += x

        return y


if __name__ == "__main__":
    # 创建一个形状为 [batch_size, channels, height, width] 的虚拟输入张量
    input = torch.randn(2, 128, 40, 40)
    ca = CA(128, 128)
    output1 = ca(input)
    print(output1.size())

    ca_ds = CA_DSConv(128, 128, 3, 2, 2)
    ds_ca = DSConv_CA(128, 128, 3, 2, 2)
    output2 = ca_ds(input)
    output3 = ds_ca(input)
    print(output2.size())
    print(output3.size())

    ca_spd = CA_SPDConv(128, 128, 3, 2, 2)
    spd_ca = SPDConv_CA(128, 128, 3, 2, 2)
    output4 = spd_ca(input)
    output5 = ca_spd(input)
    print(output4.size())
    print(output5.size())

    ca_hwd = CA_HWD(128, 128, 3, 2, 2)
    hwd_ca = HWD_CA(128, 128, 3, 2, 2)
    output6 = hwd_ca(input)
    output7 = ca_hwd(input)
    print(output6.size())
    print(output7.size())

    ca_conv = CA_Conv(128, 128, 3, 2, 2)
    conv_ca = Conv_CA(128, 128, 3, 2, 2)
    output8 = conv_ca(input)
    output9 = ca_conv(input)
    print(output8.size())
    print(output9.size())
