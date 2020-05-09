import torch.nn as nn
from torchvision.models import resnet18
import torch
class FCN(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()

        self.base_model = resnet18(pretrained=True)
        layers = list(self.base_model.children())
        conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.layer1 = nn.Sequential(
            conv1,
            *layers[1:5]
        )
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.layer2 = layers[5]
        self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.layer3 = layers[6]
        self.upsample3 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.layer4 = layers[7]
        self.upsample4 = nn.Upsample(scale_factor=32, mode='bilinear')
        self.conv1k = nn.Conv2d(64 + 128 + 256 + 512, n_classes, 1)
    def forward(self,x):
        x =self.layer1(x)
        up1 = self.upsample1(x)
        x = self.layer2(x)
        up2 = self.upsample2(x)
        x = self.layer3(x)
        up3 = self.upsample3(x)
        x = self.layer4(x)
        up4 = self.upsample4(x)

        merge = torch.cat([up1, up2, up3, up4], dim=1)
        out = self.conv1k(merge)
        return out
def test():
    model = FCN()
    x = model(torch.rand(1,1,224,224))
    print(x.size())
