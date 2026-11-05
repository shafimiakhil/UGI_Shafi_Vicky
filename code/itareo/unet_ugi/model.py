"""U-Net with a torchvision ResNet-18 encoder (ImageNet weights) for 4-band input R, G, B, NIR.

conv1: the pretrained RGB filters are kept for input channels 1-3 (R, G, B mapped to ImageNet R, G, B) and the NIR
filter is set to their mean (CLAUDE.md 12.3 e). The decoder upsamples bilinearly and concatenates the four skip levels
(1/16, 1/8, 1/4, 1/2), as in the D1 decoder of itareo.models, then a 1 x 1 head gives K class logits at input size.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

ENC = (64, 64, 128, 256, 512)


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def resnet18_4band(pretrained=True):
    """torchvision ResNet-18 with a 4-channel conv1: channels 1-3 = pretrained RGB filters, channel 4 = their mean."""
    w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    r = torchvision.models.resnet18(weights=w)
    old = r.conv1.weight.data
    conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        conv1.weight.copy_(torch.cat([old, old.mean(1, keepdim=True)], 1))
    r.conv1 = conv1
    return r


class UNetResNet18(nn.Module):
    def __init__(self, n_classes, pretrained=True, dec=(256, 128, 64, 32, 16)):
        super().__init__()
        self.enc = resnet18_4band(pretrained)
        self.blocks = nn.ModuleList()
        cin = ENC[4]
        for i, skip in enumerate(reversed(ENC[:4])):
            self.blocks.append(conv_block(cin + skip, dec[i]))
            cin = dec[i]
        self.final = conv_block(cin, dec[4])
        self.head = nn.Conv2d(dec[4], n_classes, 1)

    def features(self, x):
        e = self.enc
        x0 = e.relu(e.bn1(e.conv1(x)))      # 1/2
        x1 = e.layer1(e.maxpool(x0))        # 1/4
        x2 = e.layer2(x1)                   # 1/8
        x3 = e.layer3(x2)                   # 1/16
        x4 = e.layer4(x3)                   # 1/32
        return [x0, x1, x2, x3, x4]

    def forward(self, x):
        f = self.features(x)
        y = f[4]
        for blk, skip in zip(self.blocks, reversed(f[:4])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = blk(torch.cat([y, skip], 1))
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.final(y))


def conv1_selftest():
    """Channels 1-3 of conv1 equal torchvision's pretrained conv1 exactly; channel 4 equals their mean."""
    ref = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).conv1.weight
    m = UNetResNet18(7, pretrained=True)
    w = m.enc.conv1.weight
    rgb_equal = bool(torch.equal(w[:, :3], ref))
    nir_is_mean = bool(torch.allclose(w[:, 3], ref.mean(1), atol=0, rtol=0))
    other_equal = bool(torch.equal(m.enc.layer4[1].conv2.weight, torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).layer4[1].conv2.weight))
    return {"conv1_channels_1_3_equal_torchvision": rgb_equal, "conv1_channel_4_equals_mean_of_1_3": nir_is_mean,
            "layer4_equal_torchvision": other_equal, "conv1_shape": list(w.shape)}
