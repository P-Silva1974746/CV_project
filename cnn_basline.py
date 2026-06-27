import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):

    def __init__(self, num_classes: int = 1):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(169 * 8 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise conv (spatial) followed by pointwise conv (channel mixing).
    """
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride, padding=padding,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class SqueezeExcitation(nn.Module):
    """
    Channel-wise attention: squeeze global spatial info.
    Ratio controls the bottleneck size of the two FC layers.
    """
    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        squeezed = max(1, channels // ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # (B, C, 1, 1)
            nn.Flatten(),                      # (B, C)
            nn.Linear(channels, squeezed, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(squeezed, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class ResidualDSBlock(nn.Module):
    """
    One residual block using depthwise separable convolutions + SE attention.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, se_ratio: int = 4):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels,
                                             stride=stride, padding=1)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels,
                                             stride=1, padding=1)
        self.se = SqueezeExcitation(out_channels, ratio=se_ratio)

        # Projection shortcut when spatial size or channel count changes
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = out + self.shortcut(x)   # residual addition
        return F.relu6(out, inplace=True)


class BaselineCNN(nn.Module):
    """
    Compact CNN baseline, with residual connections, depthwise convulotions, and squeeze excitation.
    """

    def __init__(self, num_classes: int = 1,
                 se_ratio: int = 4,
                 dropout: float = 0.4):
        super().__init__()


        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )


        self.stage1 = nn.Sequential(
            ResidualDSBlock(32,  64,  stride=2, se_ratio=se_ratio),
            ResidualDSBlock(64,  64,  stride=1, se_ratio=se_ratio),
        )
        self.stage2 = nn.Sequential(
            ResidualDSBlock(64,  128, stride=2, se_ratio=se_ratio),
            ResidualDSBlock(128, 128, stride=1, se_ratio=se_ratio),
        )
        self.stage3 = nn.Sequential(
            ResidualDSBlock(128, 256, stride=2, se_ratio=se_ratio),
            ResidualDSBlock(256, 256, stride=1, se_ratio=se_ratio),
        )


        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    #--------------------------------- Weight initialisation ---------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    #--------------------------------- Weight initialisation ---------------------------------


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)     # (B, 32,  64, 64)
        x = self.stage1(x)   # (B, 64,  32, 32)
        x = self.stage2(x)   # (B, 128, 16, 16)
        x = self.stage3(x)   # (B, 256,  8,  8)
        x = self.head(x)     # (B, num_classes)
        return x



if __name__ == "__main__":
    model = BaselineCNN(num_classes=1)
    dummy = torch.randn(4, 3, 416, 416)
    out = model(dummy)
    print(f"Output shape : {out.shape}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params : {total:,}")
    print(f"Trainable    : {trainable:,}")