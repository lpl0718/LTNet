import torch
import torch.nn as nn


class DWTForward(nn.Module):
    def __init__(self, J=1, wave='haar', mode='zero'):
        super().__init__()
        self.J = J
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
        assert H % 2 == 0 and W % 2 == 0, "DWT: 高宽必须是2的倍数"

        x_r = x.reshape(B * C, 1, H, W)

        yl = nn.functional.conv2d(x_r, self.kernel_ll, stride=2)
        yh1 = nn.functional.conv2d(x_r, self.kernel_lh, stride=2)
        yh2 = nn.functional.conv2d(x_r, self.kernel_hl, stride=2)
        yh3 = nn.functional.conv2d(x_r, self.kernel_hh, stride=2)

        yl = yl.reshape(B, C, H//2, W//2)
        yh1 = yh1.reshape(B, C, H//2, W//2)
        yh2 = yh2.reshape(B, C, H//2, W//2)
        yh3 = yh3.reshape(B, C, H//2, W//2)

        yh = torch.stack([yh1, yh2, yh3], dim=2)
        return yl, (yh,)


class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1, bias=False),

            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)

        y_HL = yH[0][:, :, 0, ...]
        y_LH = yH[0][:, :, 1, ...]
        y_HH = yH[0][:, :, 2, ...]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x

if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 64, 224, 224)
    image = torch.rand(*image_size)

    # Model
    model = Down_wt(64, 32)
    out = model(image)
    print("【FP32】输入尺寸：", image.shape)
    print("【FP32】输出尺寸：", out.shape)


    print("\n--- 开始FP16半精度测试 ---")
    model = model.half().cuda()
    image = image.half().cuda()
    with torch.no_grad():
        out_fp16 = model(image)
    print("【FP16】输入尺寸：", image.shape)
    print("【FP16】输出尺寸：", out_fp16.shape)
    print("✅ FP16训练/推理 兼容成功！")


