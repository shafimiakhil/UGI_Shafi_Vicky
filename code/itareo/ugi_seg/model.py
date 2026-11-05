"""U-Net with a binary UGI head (channel 0) and a type-group head (channels 1-5), 6-band Neo input (R, G, B, NIR, RE, DB).

conv1 of the encoder has 6 input channels:
  * A1 (ImageNet): channels 1-4 as itareo.unet_ugi.model (R, G, B = torchvision's pretrained RGB filters, NIR = their
    mean); RE and DB = the mean of channels 1-4 (an extra band gets the mean of the pretrained filters, as NIR does);
  * A2 (B3 encoder): the whole encoder (conv1 to layer4, BatchNorm included) from the B3 GEOBIA land-cover checkpoint
    of the same fold and seed, whose conv1 has R, G, B, NIR (Pléiades 2017 band order); RE and DB = the mean of those
    four filters. The decoder and the heads start from scratch in both.
"""
import torch
import torch.nn as nn

from itareo.unet_ugi.model import UNetResNet18

N_OUT = 6


def widen_conv1(conv4):
    """4-channel conv1 -> 6-channel (R, G, B, NIR, RE, DB): channels 1-4 kept, 5 and 6 = mean of 1-4."""
    w = conv4.weight.data
    conv = nn.Conv2d(6, w.shape[0], kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        m = w.mean(1, keepdim=True)
        conv.weight.copy_(torch.cat([w, m, m], 1))
    return conv


def build(variant, b3_checkpoint=None):
    if variant == "imagenet":
        m = UNetResNet18(N_OUT, pretrained=True)
    elif variant == "b3":
        m = UNetResNet18(N_OUT, pretrained=False)
        ck = torch.load(b3_checkpoint, map_location="cpu", weights_only=False)
        enc = {k[len("enc."):]: v for k, v in ck["model"].items() if k.startswith("enc.")}
        missing, unexpected = m.enc.load_state_dict(enc, strict=False)
        assert not unexpected and all(k.startswith("fc.") for k in missing), (missing, unexpected)
    else:
        raise ValueError(variant)
    m.enc.conv1 = widen_conv1(m.enc.conv1)
    return m


def selftest(b3_checkpoint=None):
    out = {}
    m = build("imagenet")
    w = m.enc.conv1.weight
    out["imagenet_conv1_shape"] = list(w.shape)
    out["imagenet_re_db_equal_mean_1_4"] = bool(torch.equal(w[:, 4], w[:, :4].mean(1)) and torch.equal(w[:, 5], w[:, :4].mean(1)))
    x = torch.randn(2, 6, 128, 128)
    out["output_shape"] = list(m(x).shape)
    if b3_checkpoint:
        ck = torch.load(b3_checkpoint, map_location="cpu", weights_only=False)["model"]
        m2 = build("b3", b3_checkpoint)
        out["b3_conv1_channels_1_4_equal"] = bool(torch.equal(m2.enc.conv1.weight[:, :4], ck["enc.conv1.weight"]))
        out["b3_layer4_equal"] = bool(torch.equal(m2.enc.layer4[1].conv2.weight, ck["enc.layer4.1.conv2.weight"]))
        out["b3_bn_running_mean_equal"] = bool(torch.equal(m2.enc.bn1.running_mean, ck["enc.bn1.running_mean"]))
    return out
