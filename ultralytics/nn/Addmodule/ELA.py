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


class ELA(nn.Module):
    def __init__(self, in_channels, out_channels, phi="T", *args, **kwargs):
        super().__init__()
        """
        ELA-T 和 ELA-B 设计为轻量级，非常适合网络层数较少或轻量级网络的 CNN 架构
        ELA-B 和 ELA-S 在具有更深结构的网络上表现最佳
        ELA-L 特别适合大型网络。
        """
        try:
            Kernel_size = {"T": 5, "B": 7, "S": 5, "L": 7}[phi]
            groups = {"T": in_channels, "B": in_channels, "S": in_channels // 8, "L": in_channels // 8}[phi]
            num_groups = {"T": 32, "B": 16, "S": 16, "L": 16}[phi]
        except KeyError:
            Kernel_size = 5
            groups = in_channels
            num_groups = 32

        pad = Kernel_size // 2
        self.con1 = nn.Conv1d(in_channels, in_channels, kernel_size=Kernel_size, padding=pad, groups=groups, bias=False)
        self.GN = nn.GroupNorm(num_groups, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        b, c, h, w = input.size()
        x_h = torch.mean(input, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(input, dim=2, keepdim=True).view(b, c, w)
        x_h = self.con1(x_h)  # [b,c,h]
        x_w = self.con1(x_w)  # [b,c,w]
        x_h = self.sigmoid(self.GN(x_h)).view(b, c, h, 1)  # [b, c, h, 1]
        x_w = self.sigmoid(self.GN(x_w)).view(b, c, 1, w)  # [b, c, 1, w]

        result = x_h * x_w * input

        return result


class DSConv_ELA(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1, phi="T", drop_connect_rate=0.2):
        super().__init__()
        self.has_shortcut = s == 1 and c1 == c2
        expanded_c = c1 * expansion
        # self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
        self.dw_conv = Conv(c1, expanded_c, k=k, s=s, g=c1)

        self.ela = ELA(expanded_c, expanded_c, phi)

        self.cv1 = Conv(expanded_c, c2, k=1, s=1)
        self.drop_connect_rate = drop_connect_rate
        if self.has_shortcut and drop_connect_rate > 0:
            self.dropout = DropPath(drop_connect_rate)

    def forward(self, x):

        # y = self.expansion_conv(x)
        y = self.dw_conv(x)
        y = self.ela(y)
        y = self.cv1(y)
        if self.has_shortcut:
            if self.drop_connect_rate > 0:
                y = self.dropout(y)
            y += x

        return y


class ELA_DSConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1, phi="T", drop_connect_rate=0.2):
        super().__init__()
        self.has_shortcut = s == 1 and c1 == c2
        expanded_c = c1 * expansion
        self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
        self.ela = ELA(expanded_c, expanded_c, phi)
        self.dw_conv = Conv(expanded_c, expanded_c, k=k, s=s, g=expanded_c)
        self.cv1 = Conv(expanded_c, c2, k=1, s=1)
        self.drop_connect_rate = drop_connect_rate
        if self.has_shortcut and drop_connect_rate > 0:
            self.dropout = DropPath(drop_connect_rate)

    def forward(self, x):

        y = self.expansion_conv(x)
        y = self.ela(y)
        y = self.dw_conv(y)
        y = self.cv1(y)
        if self.has_shortcut:
            if self.drop_connect_rate > 0:
                y = self.dropout(y)
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


class SPDConv_ELA(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1, phi="T", drop_connect_rate=0.2):
        super().__init__()
        self.has_shortcut = s == 1 and c1 == c2
        expanded_c = c1 * expansion
        # self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
        self.spd_conv = SPDConv(c1, expanded_c, k=1, s=1)

        self.ela = ELA(expanded_c, expanded_c, phi)
        self.cv1 = Conv(expanded_c, c2, k=1, s=1)
        self.drop_connect_rate = drop_connect_rate
        if self.has_shortcut and drop_connect_rate > 0:
            self.dropout = DropPath(drop_connect_rate)

    def forward(self, x):

        # y = self.expansion_conv(x)
        y = self.spd_conv(x)
        y = self.ela(y)
        y = self.cv1(y)

        if self.has_shortcut:
            if self.drop_connect_rate > 0:
                y = self.dropout(y)
            y += x

        return y


class ELA_SPDConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1, phi="T", drop_connect_rate=0.2):
        super().__init__()
        self.has_shortcut = s == 1 and c1 == c2
        expanded_c = c1 * expansion
        self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
        self.ela = ELA(expanded_c, expanded_c, phi)
        self.spd_conv = SPDConv(expanded_c, c2, k=1, s=1)
        # self.cv1 = Conv(expanded_c, c2, k=1, s=1)
        self.drop_connect_rate = drop_connect_rate
        if self.has_shortcut and drop_connect_rate > 0:
            self.dropout = DropPath(drop_connect_rate)

    def forward(self, x):

        y = self.expansion_conv(x)
        y = self.ela(y)
        y = self.spd_conv(y)

        if self.has_shortcut:
            if self.drop_connect_rate > 0:
                y = self.dropout(y)
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
            # nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.03),
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


class HWD_ELA(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1, phi="T", drop_connect_rate=0.2):
        super().__init__()
        self.has_shortcut = s == 1 and c1 == c2
        expanded_c = c1 * expansion
        # self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
        self.hwd = Down_wt(c1, expanded_c)

        self.ela = ELA(expanded_c, expanded_c, phi)

        self.cv1 = Conv(expanded_c, c2, k=1, s=1)
        self.drop_connect_rate = drop_connect_rate
        if self.has_shortcut and drop_connect_rate > 0:
            self.dropout = DropPath(drop_connect_rate)

    def forward(self, x):

        # y = self.expansion_conv(x)
        y = self.hwd(x)
        y = self.ela(y)
        y = self.cv1(y)
        if self.has_shortcut:
            if self.drop_connect_rate > 0:
                y = self.dropout(y)
            y += x

        return y


class ELA_HWD(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1, phi="T", drop_connect_rate=0.2):
        super().__init__()
        self.has_shortcut = s == 1 and c1 == c2
        expanded_c = c1 * expansion
        self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
        self.ela = ELA(expanded_c, expanded_c, phi)
        self.hwd = Down_wt(expanded_c, c2)
        # self.cv1 = Conv(expanded_c, c2, k=1, s=1)
        self.drop_connect_rate = drop_connect_rate
        if self.has_shortcut and drop_connect_rate > 0:
            self.dropout = DropPath(drop_connect_rate)

    def forward(self, x):

        y = self.expansion_conv(x)
        y = self.ela(y)
        y = self.hwd(y)

        if self.has_shortcut:
            if self.drop_connect_rate > 0:
                y = self.dropout(y)
            y += x

        return y


class Conv_ELA(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1, phi="T", drop_connect_rate=0.2):
        super().__init__()
        self.has_shortcut = s == 1 and c1 == c2
        expanded_c = c1 * expansion
        # self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
        self.conv = Conv(c1, expanded_c, k=k, s=s)

        self.ela = ELA(expanded_c, expanded_c, phi)

        self.cv1 = Conv(expanded_c, c2, k=1, s=1)
        self.drop_connect_rate = drop_connect_rate
        if self.has_shortcut and drop_connect_rate > 0:
            self.dropout = DropPath(drop_connect_rate)

    def forward(self, x):

        # y = self.expansion_conv(x)
        y = self.conv(x)
        y = self.ela(y)
        y = self.cv1(y)
        if self.has_shortcut:
            if self.drop_connect_rate > 0:
                y = self.dropout(y)
            y += x

        return y


class ELA_Conv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1, phi="T", drop_connect_rate=0.2):
        super().__init__()
        self.has_shortcut = s == 1 and c1 == c2
        expanded_c = c1 * expansion
        self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
        self.ela = ELA(expanded_c, expanded_c, phi)
        self.conv = Conv(expanded_c, c2, k=k, s=s)
        # self.cv1 = Conv(expanded_c, c2, k=1, s=1)
        self.drop_connect_rate = drop_connect_rate
        if self.has_shortcut and drop_connect_rate > 0:
            self.dropout = DropPath(drop_connect_rate)

    def forward(self, x):

        y = self.expansion_conv(x)
        y = self.ela(y)
        y = self.conv(y)

        if self.has_shortcut:
            if self.drop_connect_rate > 0:
                y = self.dropout(y)
            y += x

        return y


if __name__ == "__main__":
    # 创建一个形状为 [batch_size, channels, height, width] 的虚拟输入张量
    input = torch.randn(2, 128, 40, 40)
    ela = ELA(128, 128, phi="T")
    output1 = ela(input)
    print(output1.size())

    ela_ds = ELA_DSConv(128, 128, 3, 2, 2, "B")
    ds_ela = DSConv_ELA(128, 128, 3, 2, 2, "B")
    output2 = ela_ds(input)
    output3 = ds_ela(input)
    print(output2.size())
    print(output3.size())

    ela_spd = ELA_SPDConv(128, 128, 3, 2, 2)
    spd_ela = SPDConv_ELA(128, 128, 3, 2, 2)
    output4 = spd_ela(input)
    output5 = ela_spd(input)
    print(output4.size())
    print(output5.size())

    ela_hwd = ELA_HWD(128, 128, 3, 2, 2)
    hwd_ela = HWD_ELA(128, 128, 3, 2, 2)
    output6 = hwd_ela(input)
    output7 = ela_hwd(input)
    print(output6.size())
    print(output7.size())

    ela_conv = ELA_Conv(128, 128, 3, 2, 2)
    conv_ela = Conv_ELA(128, 128, 3, 2, 2)
    output8 = conv_ela(input)
    output9 = ela_conv(input)
    print(output8.size())
    print(output9.size())
