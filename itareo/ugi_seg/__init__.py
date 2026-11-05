"""ugi_seg: UGI segmentation (CLAUDE.md 13, family A of the design note "Multi-sensor UGI mapping, research design").

One U-Net (torchvision ResNet-18 encoder, decoder of itareo.unet_ugi) with two heads on the same decoder output:
channel 0 = binary UGI logit, channels 1-5 = type-group logits (Road greens, Squares, Parks and green areas, Water
corridors, Rail verges). Labels from derived/ugi_labels/ (inventory, smaller polygon wins overlaps).
  * binary head: UGI = 1, background = 0 (certain negative, treatment D1), the 1 m edge band ignored;
  * group head: only UGI pixels with a defined group (NO DEFINIDO and background ignored), edge band ignored.
A1: six Neo 2022 bands from ImageNet weights; A2: the same, encoder from the B3 GEOBIA land-cover model of the same
fold and seed. Reuses itareo.unet_ugi for folds, internal-validation blocks, buffer, sampler and the decoder.
"""
