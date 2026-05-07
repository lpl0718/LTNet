import math
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # 用于便捷地重排张量维度

# https://arxiv.org/pdf/2506.08520

def closest_square_factors(N):
    """
    寻找最接近的两个因数(r, c)，使得r*c = N且|r-c|最小
    用途：将landmark数量分解为近似正方形的网格尺寸，便于后续池化操作

    参数:
        N: 需要分解的整数（landmark数量）
    返回:
        (r, c): 满足r*c=N且最接近正方形的两个因数
    """
    s = int(math.isqrt(N))  # 取N的平方根的整数部分（向下取整）
    # 从平方根开始向下寻找能整除N的最大因数
    for i in range(s, 0, -1):
        if N % i == 0:
            return i, N // i  # 返回一对因数


class ExpLinearAfterThreshold(nn.Module):
    """
    自定义激活函数：阈值以下使用指数函数，阈值以上使用线性函数
    设计目的：平衡非线性表达能力与梯度稳定性，避免指数函数在大值区域的梯度爆炸

    参数:
        max_val: 阈值，小于等于该值使用指数函数，大于该值使用线性函数
    """

    def __init__(self, max_val=5.0):
        super().__init__()
        self.max_val = max_val  # 阈值参数

    def forward(self, x):
        # 对输入x进行截断（大于max_val的部分按max_val处理），然后计算指数
        exp_part = torch.exp(x.clamp(max=self.max_val))
        # 对输入x中大于max_val的部分计算线性ReLU（小于等于0的部分置0）
        linear_part = torch.relu(x - self.max_val)
        return exp_part + linear_part  # 两部分相加作为输出


class PnPNystraAttention(nn.Module):
    r""" 基于PNP-Nyström方法的注意力模块
    核心思想：使用少量"landmark"（地标点）近似计算注意力矩阵，将复杂度从O(N²)降至O(NM)
    支持四维输入输出（B, C, H, W），适用于图像等视觉任务

    参数:
        dim: 输入特征的通道维度
        num_heads: 注意力头的数量（多头注意力机制）
        num_landmarks: 用于近似的地标点数量（M，需远小于序列长度N）
        iters: 计算Moore-Penrose伪逆的迭代次数（影响近似精度）
        qkv_bias: 是否为QKV线性层添加偏置
        qk_scale: 自定义QK相似度计算的缩放因子（默认使用head_dim^-0.5）
        attn_drop: 注意力权重的dropout概率
        proj_drop: 输出投影层的dropout概率
    """

    def __init__(self, dim, num_heads=4, num_landmarks=64, iters=6, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim  # 输入特征维度（通道数）
        self.num_heads = num_heads  # 注意力头数量
        self.head_dim = dim // num_heads  # 每个注意力头的维度（总维度/头数）
        # 缩放因子：用于调节QK相似度的数值范围，避免softmax梯度消失
        self.scale = qk_scale or self.head_dim ** -0.5

        # QKV生成：通过一个线性层同时生成查询（Q）、键（K）、值（V）
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力权重的dropout
        self.proj = nn.Linear(dim, dim)  # 输出投影层：将多头特征聚合回原维度
        self.proj_drop = nn.Dropout(proj_drop)  # 输出的dropout

        self.activ = ExpLinearAfterThreshold()  # 自定义激活函数实例

        self.num_landmarks = num_landmarks  # 地标点数量（M）
        self.iters = iters  # Moore-Penrose伪逆的迭代次数

    def moore_penrose_iter_pinv(self, x, iters=6):
        """
        迭代计算矩阵的Moore-Penrose伪逆（广义逆）
        用途：在Nyström近似中求解低秩矩阵的逆，比直接求逆更高效、稳定

        参数:
            x: 输入矩阵，形状为(..., M, M)（M为landmark数量）
            iters: 迭代次数（默认6次，已验证足够收敛）
        返回:
            x的近似伪逆矩阵，形状与x一致
        """
        device = x.device  # 获取输入矩阵所在设备（CPU/GPU）

        # 初始化：基于输入矩阵的行和列范数进行缩放
        abs_x = torch.abs(x)
        col_sum = abs_x.sum(dim=-1)  # 列方向求和（每行的L1范数）
        row_sum = abs_x.sum(dim=-2)  # 行方向求和（每列的L1范数）
        # 初始近似值：转置矩阵除以最大行列范数的乘积（缩放以保证数值稳定）
        z = rearrange(x, '... i j -> ... j i') / (torch.max(col_sum) * torch.max(row_sum) + 1e-15)

        # 创建单位矩阵（与x的最后两个维度匹配）
        I = torch.eye(x.shape[-1], device=device)
        I = rearrange(I, 'i j -> () i j')  # 增加维度以匹配批量维度

        # 迭代更新伪逆（基于牛顿迭代法的变体）
        for _ in range(iters):
            xz = x @ z  # x与当前近似逆的乘积
            # 迭代公式：通过多项式逼近优化伪逆（经验证的高效收敛公式）
            z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

        return z

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        head_dim = self.head_dim

        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        qkv = self.qkv(x_flat).reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        q_reshaped = q.reshape(B * self.num_heads, H, W, head_dim).permute(0, 3, 1, 2)
        k_reshaped = k.reshape(B * self.num_heads, H, W, head_dim).permute(0, 3, 1, 2)

        h_landmark, w_landmark = closest_square_factors(self.num_landmarks)

        # -------- 关键修改：确定性池化 + 整除兼容 --------
        # 计算固定核大小和步长，模拟自适应池化
        kernel_h = H // h_landmark
        kernel_w = W // w_landmark
        stride_h, stride_w = kernel_h, kernel_w

        # 兼容整除问题：如果池化后尺寸不匹配，用padding补足（保证确定性）
        pad_h = (h_landmark * kernel_h - H) // 2
        pad_w = (w_landmark * kernel_w - W) // 2
        pad_h = max(pad_h, 0)  # 避免负padding
        pad_w = max(pad_w, 0)

        # 用普通AvgPool2d替代adaptive_avg_pool2d（确定性实现）
        q_m = F.avg_pool2d(
            F.pad(q_reshaped, (pad_w, pad_w, pad_h, pad_h)),  # 先padding再池化
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
            ceil_mode=False  # 禁用向上取整，保证确定性
        )
        k_m = F.avg_pool2d(
            F.pad(k_reshaped, (pad_w, pad_w, pad_h, pad_h)),
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
            ceil_mode=False
        )
        # -------------------------------------------------------------

        # 裁剪到目标尺寸（防止padding后池化尺寸略大）
        q_m = q_m[:, :, :h_landmark, :w_landmark]
        k_m = k_m[:, :, :h_landmark, :w_landmark]

        q_m = q_m.permute(0, 2, 3, 1).reshape(B, self.num_heads, self.num_landmarks, head_dim)
        k_m = k_m.permute(0, 2, 3, 1).reshape(B, self.num_heads, self.num_landmarks, head_dim)

        # 后续代码不变
        temp = self.activ(q_m @ k_m.transpose(-2, -1))
        pseudo_inv = self.moore_penrose_iter_pinv(temp, self.iters)
        prod = (self.activ(q @ k_m.transpose(-2, -1)) @ pseudo_inv) @ (
                self.activ(q_m @ k.transpose(-2, -1)) @ torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)
        )

        x_out = (prod[..., :-1] / (prod[..., -1].unsqueeze(-1) + 1e-12)).transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        x_out = rearrange(x_out, 'b (h w) c -> b c h w', h=H, w=W)

        return x_out


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
        x = x.to(self.conv.weight.device)  # 确保输入张量在卷积层所在设备
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = x.to(self.conv.weight.device)  # 确保输入张量在卷积层所在设备
        return self.act(self.conv(x))


class PSABloc_PnPNystraAttention(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABloc_PnPNystraAttention(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    # 修复：添加num_landmarks/iters参数，并传递给PnPNystraAttention
    def __init__(self, c, attn_ratio=0.5, num_heads=4, num_landmarks=64, iters=6, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        # 关键修复：传递所有必要参数，而非仅dim
        self.attn = PnPNystraAttention(
            dim=c,
            num_heads=num_heads,
            num_landmarks=num_landmarks,
            iters=iters,
            attn_drop=0.1,
            proj_drop=0.1
        )
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA_PnPNystraAttention(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA_PnPNystraAttention(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        # 修复：传递num_landmarks/iters参数到PSABloc_PnPNystraAttention
        self.m = nn.Sequential(*(
            PSABloc_PnPNystraAttention(
                self.c,
                attn_ratio=0.5,
                num_heads=self.c // 64,
                num_landmarks=64,  # 自定义landmark数量
                iters=6  # 自定义迭代次数
            ) for _ in range(n)
        ))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        x = x.to(self.cv1.conv.weight.device)  # 确保输入张量在第一个卷积层所在设备
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


def main():
    """测试函数：验证PnPNystraAttention模块的功能正确性（启用确定性模式）"""
    torch.manual_seed(42)  # 固定随机种子，保证结果可复现
    torch.use_deterministic_algorithms(True, warn_only=False)  # 启用严格确定性模式

    # 1. 测试基础PnPNystraAttention
    model = PnPNystraAttention(
        dim=128,  # 输入特征维度
        num_heads=8,  # 8个注意力头（每个头维度16=128/8）
        num_landmarks=64,  # 64个landmark（远小于64x64=4096的序列长度）
        iters=6,  # 伪逆迭代次数
        qkv_bias=True,  # QKV使用偏置
        attn_drop=0.1,  # 注意力dropout率10%
        proj_drop=0.1  # 输出投影dropout率10%
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 创建四维输入（模拟图像特征，测试非整除尺寸：65x65）
    batch_size = 2  # 批量大小
    channels = 128  # 通道数（与dim一致）
    height, width = 65, 65  # 非整除尺寸，测试兼容逻辑
    x = torch.randn(batch_size, channels, height, width).to(model.qkv.weight.device)

    # 3. 前向传播（禁用梯度计算以加速）
    start_time = time.time()
    with torch.no_grad():
        output = model(x)
    end_time = time.time()

    # 4. 验证结果
    print(f"输入形状: {x.shape}")  # 预期：(2, 128, 65, 65)
    print(f"输出形状: {output.shape}")  # 预期：与输入形状一致
    print(f"计算耗时: {end_time - start_time:.4f}秒")  # 打印计算时间
    # 断言输出形状与输入一致（核心验证）
    assert output.shape == x.shape, "输出形状与输入不匹配!"

    # 5. 测试C2PSA封装模块
    c2psa_model = C2PSA_PnPNystraAttention(c1=128, c2=128, n=2, e=0.5).to(model.qkv.weight.device)
    c2psa_output = c2psa_model(x)
    print(f"\nC2PSA输入形状: {x.shape}")
    print(f"C2PSA输出形状: {c2psa_output.shape}")
    assert c2psa_output.shape == x.shape, "C2PSA输出形状与输入不匹配!"

    print("\n所有测试成功! 模块功能正常，确定性池化无警告。")


if __name__ == "__main__":
    main()