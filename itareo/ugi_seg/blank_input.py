"""D2 of CLAUDE.md 13.8: a plugin of itareo.ugi_seg.oof whose window gives the added input of a fusion model at its no-data value, whatever the position: every channel 0, which is what ps_branch.ps_tensor and tex_branch.tex_tensor give where the input does not exist (normalised values 0, availability and masks 0). How often training holds this state differs. E1b: no date on 4.83 to 11.93 % of the training cells of folds 1 to 4, none inside sq8km. E2: both masks 0 on 0.0017 to 0.0045 % of training pixels, the NDVI-texture mask 0 on at most 678 pixels of any tile, so for E2 a whole window in this state lies outside training. The model is the trained one, unchanged. Config: model.plugin {module: itareo.ugi_seg.blank_input, shape: [channels, height, width]}. Used by scripts/diag_d2.py."""
import torch


def window(cfg, fold, ck, region):
    z = torch.zeros(tuple(cfg["model"]["plugin"]["shape"]))

    def blank(row0, col0):
        return z
    return blank
