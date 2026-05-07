import torch
import torch.nn as nn

__all__ = ["SPD", "SPDConv", "SPDConv2"]


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


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
        self.c1 = c1
        self.c2 = c1 * 4

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


#         size_tensor = x.size()
#         return torch.cat([x[...,0:size_tensor[2]//2,0:size_tensor[3]//2],
#                          x[...,0:size_tensor[2]//2,size_tensor[3]//2:],
#                          x[...,size_tensor[2]//2:,0:size_tensor[3]//2],
#                          x[...,size_tensor[2]//2:,size_tensor[3]//2:]  ],1)


class SPD9(nn.Module):
    """"""

    def __init__(self, c1, c2=None, d=1):
        super().__init__()
        self.d = d
        self.c1 = c1
        self.step = 2
        self.c2 = c1 * 9 if c2 is None else c2
        self.kernel_size = 3

    def forward(self, x):

        _B, _C, ori_H, ori_W = x.size()

        target_h = ori_H // 2
        target_w = ori_W // 2

        pad_top = self.kernel_size // 2
        pad_left = self.kernel_size // 2

        x = nn.ConstantPad2d((pad_left, 0, pad_top, 0), 0.0)(x)

        x00 = x[..., 0 :: self.step, 0 :: self.step]
        x01 = x[..., 0 :: self.step, 1 :: self.step]
        x02 = x[..., 0 :: self.step, 2 :: self.step]
        x10 = x[..., 1 :: self.step, 0 :: self.step]
        x11 = x[..., 1 :: self.step, 1 :: self.step]
        x12 = x[..., 1 :: self.step, 2 :: self.step]
        x20 = x[..., 2 :: self.step, 0 :: self.step]
        x21 = x[..., 2 :: self.step, 1 :: self.step]
        x22 = x[..., 2 :: self.step, 2 :: self.step]

        grids = [x00, x01, x02, x10, x11, x12, x20, x21, x22]
        unified_grids = []
        for grid in grids:
            crop_h = min(grid.shape[2], target_h)
            crop_w = min(grid.shape[3], target_w)
            grid_cropped = grid[..., :crop_h, :crop_w]
            unified_grids.append(grid_cropped)

        x = torch.cat(unified_grids, dim=1)
        return x


class SPDConv2(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.d = d
        self.out = c2
        self.spd9 = SPD9(c1)
        self.conv = Conv(c1 * 9, c2, k, s, p=p, g=g, d=d, act=act)

    def forward(self, x):
        # 验证输入尺寸
        ori_h, ori_w = x.shape[2], x.shape[3]
        x = self.spd9(x)
        x = self.conv(x)
        assert x.shape[2] <= ori_h // 2 and x.shape[3] <= ori_w // 2, (
            f"输出尺寸{x.shape[2:]}超过原始半尺寸({ori_h // 2}, {ori_w // 2})"
        )
        return x


if __name__ == "__main__":
    image_size = (1, 13, 240, 240)
    image = torch.rand(*image_size)

    # Model
    test = SPDConv(13, 33, k=3, s=2)

    out = test(image)
    print(out.size())
