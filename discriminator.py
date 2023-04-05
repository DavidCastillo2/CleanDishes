import torch
import torch.nn as nn
from config import IMAGE_SIZE_Y, IMAGE_SIZE_X


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, inChannels=3, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.initial = nn.Sequential(
            nn.Conv2d(
                inChannels * 2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        inChannels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(inChannels, feature, stride=1 if feature == features[-1] else 2),
            )
            inChannels = feature

        layers.append(
            nn.Conv2d(
                inChannels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            ),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x


def test():
    x = torch.randn((1, 3, IMAGE_SIZE_X, IMAGE_SIZE_Y))
    y = torch.randn((1, 3, IMAGE_SIZE_X, IMAGE_SIZE_Y))
    model = Discriminator(inChannels=3)
    preds = model(x, y)
    print(model)
    print(preds.shape)


if __name__ == "__main__":
    test()


