import sys
from pathlib import Path

import torch
import torch.nn as nn

# 自动加入根路径，永远不炸
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent.parent.parent))

# 永远不报错的导入
try:
    from .SPDConv import SPDConv
except:
    from SPDConv import SPDConv

from ultralytics.nn.modules import DWConv


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


# 三分支并行下采样模块
# Triple-Branch Parallel Downsampling Module

# class TBPD(nn.Module):#双通道下采样拼接
#
#
#
#     def __init__(self, c1, c2):
#         """Initialize Conv layer with given arguments including activation."""
#         super().__init__()
#         c_=c1 // 2
#         self.cv1 = nn.Conv2d(c_, c_, 3, 1, 1, groups=c_, dilation=1, bias=False)
#         self.spd = SPD(c_)
#         self.bn1 = nn.BatchNorm2d(4*c_)
#         self.act1 = nn.SiLU()
#         self.cv2 = Conv(4*c_,2*c_, 1, 1)
#
#         self.spdconv2=SPDConv2(c_,c_)
#         self.avg = nn.AvgPool2d(3,2,1)
#
#         self.bn2 = nn.BatchNorm2d(c_)
#         self.act2 = nn.SiLU()
#
#
#     def forward(self, x):
#
#         x=list(torch.chunk(x,2, 1))
#         y0 = self.cv1(x[0])
#         y0 = self.spd(y0)
#         y0 = self.bn1(y0)
#         y0 = self.act1(y0)
#         y0 = self.cv2(y0)
#
#         y11 = self.avg(x[1])
#         y11 = self.bn2(y11)
#         y11 = self.act2(y11)
#
#         y12= self.spdconv2(x[1])
#
#         return torch.cat([y0,y11,y12],dim=1)
#
#     def forward_fuse(self, x):
#         x=list(torch.chunk(x,2, 1))
#         y0 = self.cv1(x[0])
#         y0 = self.spd(y0)
#
#         y0 = self.act1(y0)
#         y0 = self.cv2(y0)
#
#         y11 = self.avg(x[1])
#
#         y11 = self.act2(y11)
#
#         y12= self.spdconv2(x[1])
#
#         return torch.cat([y0,y11,y12],dim=1)


# def channel_shuffle(x, groups):
#     B, C, H, W = x.size()
#     c_per_group = C // groups
#     x = x.view(B, groups, c_per_group, H, W)
#     x = torch.transpose(x, 1, 2).contiguous()
#     x = x.view(B, -1, H, W)
#     return x
#
# class ShuffleBlock(nn.Module):
#     """
#     分组卷积 + ChannelShuffle + 残差
#     可用于 stride=1 普通层 / stride=2 下采样层
#     """
#     def __init__(self, c1, c2, stride=1, groups=8):
#         super().__init__()
#         self.groups = groups
#         self.stride = stride
#
#         # 中间通道数（保持轻量化）
#         c_mid = 2*c2 if stride == 1 else c2
#
#         # 1x1 分组卷积
#         self.conv1 = nn.Conv2d(c1, c_mid, 1, groups=groups, bias=False)
#         self.bn1 = nn.BatchNorm2d(c_mid)
#         self.act1 = nn.SiLU()  # YOLO 默认 SiLU
#
#         # 3x3 DW/分组卷积 + 下采样
#         self.conv2 = nn.Conv2d(c_mid, c_mid, 3, stride=stride, padding=1, groups=c_mid, bias=False)
#         self.bn2 = nn.BatchNorm2d(c_mid)
#         self.act2 = nn.SiLU()
#
#         # 1x1 恢复通道
#         self.conv3 = nn.Conv2d(c_mid, c2, 1, bias=False)
#         self.bn3 = nn.BatchNorm2d(c2)
#         self.act3 = nn.SiLU()
#
#         # 下采样时的 shortcut 匹配
#         self.shortcut = nn.Sequential()
#         if stride != 1 or c1 != c2:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(c1, c2, 1, stride=stride, bias=False),
#                 nn.BatchNorm2d(c2)
#             )
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.act1(out)
#         out = channel_shuffle(out, self.groups)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.act2(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         out = out + self.shortcut(x)
#         out = self.act3(out)
#         return out


# 双分支并行下采样模块（Dual-Branch Parallel Downsampling, DBPD）
class DBPD3(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        self.dwconv = DWConv(c1, c1, 3, 1)
        self.spdconv = SPDConv(c1, 2 * c1)
        self.conv = Conv(c1, c1, 3, 2)

    def forward(self, x):
        y0 = self.dwconv(x)
        y0 = self.spdconv(y0)
        y1 = self.conv(x)
        return torch.cat([y0, y1], dim=1)


# 双分支并行下采样模块（Dual-Branch Parallel Downsampling, DBPD）
class DBPD2(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        self.conv = Conv(c1, c1, 3, 2)
        self.avgpool = nn.AvgPool2d(3, 2, 1)
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.SiLU()

    def forward(self, x):
        y0 = self.conv(x)
        y1 = self.avgpool(x)
        y1 = self.bn(y1)
        y1 = self.act(y1)
        return torch.cat([y0, y1], dim=1)

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        y0 = self.conv(x)
        y1 = self.avgpool(x)
        y1 = self.act(y1)
        return torch.cat([y0, y1], dim=1)


if __name__ == "__main__":
    input = torch.randn(2, 8, 320, 320)
    dbpd = DBPD2(8, 16)
    tbpd = DBPD3(8, 24)
    output1 = dbpd(input)

    print(output1.size())

    output2 = tbpd(input)

    print(output2.size())
