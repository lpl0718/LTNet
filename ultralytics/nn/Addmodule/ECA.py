import torch
from torch import nn
from torch.nn.parameter import Parameter

from ultralytics.nn.modules import \
    DWConv

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


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize

    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ECA(nn.Module):
   """构造ECA模块
   参数:
       channel: 输入特征图的通道数
       k_size: 核心大小的自适应选择
   """
   def __init__(self, C1,C2 ,k_size=3):
       super(ECA, self).__init__()
       self.avg_pool = nn.AdaptiveAvgPool2d(1)
       self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
       self.sigmoid = nn.Sigmoid()
   def forward(self, x):
       # x: 输入特征，形状为[b, c, h, w]
       b, c, h, w = x.size()
       # 在全局空间信息上的特征描述符
       y = self.avg_pool(x)
       # ECA模块的两个不同分支
       y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
       # 多尺度信息融合
       y = self.sigmoid(y)
       return  x * y.expand_as(x)


class DSConv_ECA(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1):
        super().__init__()
        self.has_shortcut = (s == 1 and c1 == c2)
        expanded_c = c1 * expansion
        self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
        self.dw_conv = Conv(expanded_c, expanded_c, k=k, s=s, g=expanded_c)

        self.eca = ECA(expanded_c,expanded_c)

        self.cv1 = Conv(expanded_c, c2, k=1, s=1)


    def forward(self, x):

        y = self.expansion_conv(x)
        y = self.dw_conv(y)
        y = self.eca(y)
        y = self.cv1(y)
        if self.has_shortcut:
            y += x

        return y

class ECA_DSConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1):
        super().__init__()
        self.has_shortcut = (s == 1 and c1 == c2)
        expanded_c = c1 * expansion
        self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
        self.eca = ECA(expanded_c,expanded_c)
        self.dw_conv = Conv(expanded_c, expanded_c, k=k, s=s, g=expanded_c)
        self.cv1 = Conv(expanded_c, c2, k=1, s=1)


    def forward(self, x):

        y = self.expansion_conv(x)
        y = self.eca(y)
        y = self.dw_conv(y)
        y = self.cv1(y)
        if self.has_shortcut:
            y += x

        return y




class SPDConv(nn.Module):
    def __init__(self, c1, c2,k=1,s=1 ,p=None ,g=1, d=1,act=True):
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


class SPDConv_ECA(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1):
        super().__init__()
        self.has_shortcut = (s == 1 and c1 == c2)
        expanded_c = c1 * expansion
        self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
        self.spd_conv = SPDConv(expanded_c, expanded_c, k=1, s=1)

        self.eca = ECA(expanded_c,expanded_c)
        self.cv1 = Conv(expanded_c, c2, k=1, s=1)




    def forward(self, x):

        y = self.expansion_conv(x)
        y = self.spd_conv(y)
        y = self.eca(y)
        y = self.cv1(y)

        if self.has_shortcut:
            y += x

        return y

class ECA_SPDConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1):
        super().__init__()
        self.has_shortcut = (s == 1 and c1 == c2)
        expanded_c = c1 * expansion
        self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
        self.eca = ECA(expanded_c,expanded_c)
        self.spd_conv = SPDConv(expanded_c, c2, k=1, s=1)
        self.cv1 = Conv(expanded_c, c2, k=1, s=1)


    def forward(self, x):

        y = self.expansion_conv(x)
        y = self.eca(y)
        y = self.spd_conv(y)


        if self.has_shortcut:
            y += x

        return y

# class DWTForward(nn.Module):
#     def __init__(self, J=1, mode='zero', wave='haar'):
#         super().__init__()
#         self.J = J
#         self.mode = mode
#         self.wave = wave

#         # ===================== 核心修改1【FP16兼容关键】=====================
#         # 1. 用nn.Parameter封装小波核，支持自动 half()/cuda() 精度/设备同步
#         # 2. 核张量写死 requires_grad=False，本身是不可训练的常量
#         # 3. 直接创建浮点型核，避免FP16下数据类型强转丢失精度
#         haar = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 2
#         h1 = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32) / 2
#         h2 = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32) / 2
#         h3 = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 2

#         self.haar = nn.Parameter(haar, requires_grad=False)
#         self.h1 = nn.Parameter(h1, requires_grad=False)
#         self.h2 = nn.Parameter(h2, requires_grad=False)
#         self.h3 = nn.Parameter(h3, requires_grad=False)

#     def forward(self, x):
#         # Haar小波变换 - 输出格式和原pytorch_wavelets库完全一致 (yL, (yH,))
#         b, c, h, w = x.size()

#         # ===================== 核心修改2【FP16兼容】=====================
#         # 校验尺寸是否是2的整数倍，FP16训练下卷积stride=2无padding，尺寸不整除必报错
#         assert h % 2 == 0 and w % 2 == 0, f"输入特征图尺寸[{h},{w}]必须是2的整数倍！当前尺寸不满足小波变换下采样要求"

#         x = x.reshape(b * c, 1, h, w)

#         # 卷积时自动拼接 [out_ch, in_ch, kh, kw] 标准卷积核格式，直接用self封装的核，自动匹配设备+精度
#         kernel = self.haar.unsqueeze(0).unsqueeze(0)
#         yL = nn.functional.conv2d(x, kernel, stride=2, padding=0)

#         y1 = nn.functional.conv2d(x, self.h1.unsqueeze(0).unsqueeze(0), stride=2, padding=0)
#         y2 = nn.functional.conv2d(x, self.h2.unsqueeze(0).unsqueeze(0), stride=2, padding=0)
#         y3 = nn.functional.conv2d(x, self.h3.unsqueeze(0).unsqueeze(0), stride=2, padding=0)

#         # 恢复维度，和原库输出格式完全对齐
#         yL = yL.reshape(b, c, yL.shape[-2], yL.shape[-1])
#         yH = torch.stack([y1, y2, y3], dim=2).reshape(b, c, 3, yL.shape[-2], yL.shape[-1])
#         return yL, (yH,)

class DWTForward(nn.Module):
    def __init__(self, J=1, wave='haar', mode='zero'):
        super().__init__()
        self.J = J
        # 直接定义4维卷积核，自动跟随模型精度/设备
        self.register_buffer('kernel_ll', torch.tensor([[[[1.0, 1.0],
                                                           [1.0, 1.0]]]], dtype=torch.float32) / 2.0)
        self.register_buffer('kernel_lh', torch.tensor([[[[1.0, -1.0],
                                                           [1.0, -1.0]]]], dtype=torch.float32) / 2.0)
        self.register_buffer('kernel_hl', torch.tensor([[[[1.0, 1.0],
                                                           [-1.0, -1.0]]]], dtype=torch.float32) / 2.0)
        self.register_buffer('kernel_hh', torch.tensor([[[[1.0, -1.0],
                                                           [-1.0, 1.0]]]], dtype=torch.float32) / 2.0)

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
        yl = yl.reshape(B, C, H//2, W//2)
        yh1 = yh1.reshape(B, C, H//2, W//2)
        yh2 = yh2.reshape(B, C, H//2, W//2)
        yh3 = yh3.reshape(B, C, H//2, W//2)

        # 输出格式对齐 pytorch_wavelets (yl, (yh,))
        yh = torch.stack([yh1, yh2, yh3], dim=2)
        return yl, (yh,)


class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1, bias=False), # 加bias=False，和BN层搭配无意义，节省显存+加速

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

class HWD_ECA(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1):
        super().__init__()
        self.has_shortcut = (s == 1 and c1 == c2)
        expanded_c = c1 * expansion
        self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
        self.hwd = Down_wt(expanded_c, expanded_c)

        self.eca = ECA(expanded_c,expanded_c)

        self.cv1 = Conv(expanded_c, c2, k=1, s=1)


    def forward(self, x):

        y = self.expansion_conv(x)
        y = self.hwd(y)
        y = self.eca(y)
        y = self.cv1(y)
        if self.has_shortcut:
            y += x

        return y

# class ECA_HWD(nn.Module):
#     def __init__(self, c1, c2, k=3, s=1, expansion=1):
#         super().__init__()
#         self.has_shortcut = (s == 1 and c1 == c2)
#         expanded_c = c1 * expansion
#         self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
#         self.eca = ECA(expanded_c,expanded_c)
#         self.hwd = Down_wt(expanded_c, c2)
#         self.cv1 = Conv(expanded_c, c2, k=1, s=1)


#     def forward(self, x):

#         y = self.expansion_conv(x)
#         y = self.eca(y)
#         y = self.hwd(y)

#         if self.has_shortcut:
#             y += x

#         return y


class ECA_HWD(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1,drop_connect_rate=0.1):
        super().__init__()
        self.has_shortcut = (s == 1 and c1 == c2)
        expanded_c = c1 * expansion
        self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
        self.eca = ECA(expanded_c,expanded_c)
        self.hwd = Down_wt(expanded_c, c2)
        # self.cv1 = Conv(expanded_c, c2, k=1, s=1)
        self.drop_connect_rate = drop_connect_rate
        if self.has_shortcut and drop_connect_rate > 0:
            self.dropout = DropPath(drop_connect_rate)

    def forward(self, x):

        y = self.expansion_conv(x)
        y = self.eca(y)
        y = self.hwd(y)

        if self.has_shortcut:
            if self.drop_connect_rate > 0:
                y = self.dropout(y)
            y += x

        return y

# class ECA_HWD(nn.Module):
#     def __init__(self, c1, c2, k=3, s=1, expansion=1):
#         super().__init__()
#         self.has_shortcut = (s == 1 and c1 == c2)
#         expanded_c = c1 * expansion
#         self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
#         self.eca = ECA(expanded_c,expanded_c)
#         self.eca2 = ECA(expanded_c,expanded_c)
#         self.hwd = Down_wt(expanded_c, c2)
#         self.cv1 = Conv(expanded_c, c2, k=1, s=1)


#     def forward(self, x):

#         y = self.expansion_conv(x)
#         y = self.eca(y)
#         y = self.eca2(y)
#         y = self.hwd(y)

#         if self.has_shortcut:
#             y += x

#         return y

class Conv_ECA(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1):
        super().__init__()
        self.has_shortcut = (s == 1 and c1 == c2)
        expanded_c = c1 * expansion
        self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
        self.conv = Conv(expanded_c, expanded_c, k=k, s=s)

        self.eca = ECA(expanded_c,expanded_c)

        self.cv1 = Conv(expanded_c, c2, k=1, s=1)


    def forward(self, x):

        y = self.expansion_conv(x)
        y = self.conv(y)
        y = self.eca(y)
        y = self.cv1(y)
        if self.has_shortcut:
            y += x

        return y

# class Conv_ECA(nn.Module):
#     def __init__(self, c1, c2, k=3, s=1, expansion=1,drop_connect_rate=0.2):
#         super().__init__()

#         expanded_c = c1 * expansion
#         # self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
#         self.conv = Conv(c1, c2, k=k,s=2 )
#         self.cv1 = Conv(c2,expanded_c,1,1)
#         self.dwconv=DWConv(expanded_c,expanded_c,3,1,expanded_c)

#         self.eca = ECA(expanded_c,expanded_c)

#         self.cv2 = Conv(expanded_c, c2, k=1, s=1)
#         self.drop_connect_rate = drop_connect_rate

#         self.dropout = DropPath(drop_connect_rate)

#     def forward(self, x):

#         # y = self.expansion_conv(x)
#         y1 = self.conv(x)
#         y2 = self.cv1(y1)
#         y2 = self.dwconv(y2)
#         y2 = self.eca(y2)
#         y2 = self.cv2(y2)


#         y2 = self.dropout(y2)
#         y = y1+y2

#         return y

class ECA_Conv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1):
        super().__init__()
        self.has_shortcut = (s == 1 and c1 == c2)
        expanded_c = c1 * expansion
        self.expansion_conv = Conv(c1, expanded_c, k=1, s=1)
        self.eca = ECA(expanded_c,expanded_c)
        self.conv = Conv(expanded_c, c2, k=k, s=s)
        self.cv1 = Conv(expanded_c, c2, k=1, s=1)


    def forward(self, x):

        y = self.expansion_conv(x)
        y = self.eca(y)
        y = self.conv(y)
        

        if self.has_shortcut:
            y += x

        return y




if __name__ == "__main__":

    # 创建一个形状为 [batch_size, channels, height, width] 的虚拟输入张量
    input = torch.randn(2, 128, 40, 40)
    eca = ECA(128,128)
    output1 = eca(input)
    print(output1.size())

    eca_ds=ECA_DSConv(128,128,3,2,2)
    ds_eca=DSConv_ECA(128,128,3,2,2)
    output2=eca_ds(input)
    output3 = ds_eca(input)
    print(output2.size())
    print(output3.size())

    eca_spd=ECA_SPDConv(128,128,3,2,2)
    spd_eca=SPDConv_ECA(128,128,3,2,2)
    output4 =spd_eca(input)
    output5 =eca_spd(input)
    print(output4.size())
    print(output5.size())


    eca_hwd=ECA_HWD(128,128,3,2,2)
    hwd_eca=HWD_ECA(128,128,3,2,2)
    output6 =hwd_eca(input)
    output7 =eca_hwd(input)
    print(output6.size())
    print(output7.size())


    eca_conv=ECA_Conv(128,128,3,2,2)
    conv_eca=Conv_ECA(128,128,3,2,2)
    output8 =conv_eca(input)
    output9 =eca_conv(input)
    print(output8.size())
    print(output9.size())
