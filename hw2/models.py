# Based on stanford-futuredata/selection-via-proxy, svp/cifar/models.py.
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, channels, width, stride=1, bottleneck=False):
        super().__init__()
        out = width * 4 if bottleneck else width
        self.bn = nn.BatchNorm2d(channels)
        kernel = 1 if bottleneck else 3
        first_stride = 1 if bottleneck else stride
        self.conv = nn.Conv2d(channels, width, kernel, first_stride, kernel // 2, bias=False)
        layers = [nn.BatchNorm2d(width), nn.ReLU()]
        second_stride = stride if bottleneck else 1
        layers.append(nn.Conv2d(width, width, 3, second_stride, padding=1, bias=False))
        if bottleneck:
            layers += [nn.BatchNorm2d(width), nn.ReLU(), nn.Conv2d(width, out, 1, bias=False)]
        self.layers = nn.Sequential(*layers)
        self.shortcut = None
        if channels != out or stride != 1:
            self.shortcut = nn.Conv2d(channels, out, 1, stride=stride, bias=False)

    def forward(self, x):
        h = F.relu(self.bn(x))
        shortcut = x if self.shortcut is None else self.shortcut(h)
        return self.layers(self.conv(h)) + shortcut


class ResNet(nn.Module):
    def __init__(self, depth=20):
        super().__init__()
        if depth not in (20, 164):
            raise ValueError("depth must be 20 or 164")
        bottleneck = depth == 164
        blocks = 18 if bottleneck else 3
        self.conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        channels = 16
        layers = []
        for stage, width in enumerate((16, 32, 64)):
            for i in range(blocks):
                stride = 2 if stage > 0 and i == 0 else 1
                layers.append(Block(channels, width, stride, bottleneck))
                channels = width * 4 if bottleneck else width
        self.layers = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(channels)
        self.fc = nn.Linear(channels, 10)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = F.relu(self.bn(self.layers(self.conv(x))))
        return self.fc(F.avg_pool2d(x, x.shape[-2:]).flatten(1))
