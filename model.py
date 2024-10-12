import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UNetConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class UNetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UNetDown, self).__init__()
        self.conv_block = UNetConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_conv = self.conv_block(x)
        x_pooled = self.pool(x_conv)
        return x_pooled, x_conv

class UNetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UNetUp, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = UNetConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        x = torch.cat([x_skip, x], dim=1)
        x = self.conv_block(x)
        return x

class UNet(nn.Module):
    def __init__(self, channels: int = 3):
        super(UNet, self).__init__()

        self.down1 = UNetDown(channels, 64)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)

        self.bottleneck = UNetConvBlock(512, 1024)

        self.up1 = UNetUp(1024, 512)
        self.up2 = UNetUp(512, 256)
        self.up3 = UNetUp(256, 128)
        self.up4 = UNetUp(128, 64)

        self.final_conv = nn.Conv2d(64, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x1_conv = self.down1(x)
        x2, x2_conv = self.down2(x1)
        x3, x3_conv = self.down3(x2)
        x4, x4_conv = self.down4(x3)

        x_bottleneck = self.bottleneck(x4)

        x_up1 = self.up1(x_bottleneck, x4_conv)
        x_up2 = self.up2(x_up1, x3_conv)
        x_up3 = self.up3(x_up2, x2_conv)
        x_up4 = self.up4(x_up3, x1_conv)

        out = self.final_conv(x_up4)
        return out
    
def test_forward_pass():
    """
    Forward Pass test to test model and verify output shape
    """ 
    model = UNet()
    x = torch.randn((16, 3, 256, 256), dtype=torch.float32)
    output = model(x)
    assert output.shape == (16, 3, 256, 256)

if __name__ == "__main__": 
    test_forward_pass()