import segmentation_models_pytorch as smp
import torch.nn as nn
import torch
from resnet3d import generate_model
from utils import normalization
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        encoder_dims[i] + encoder_dims[i - 1],
                        encoder_dims[i - 1],
                        3,
                        1,
                        1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(encoder_dims[i - 1]),
                    nn.ReLU(inplace=True),
                )
                for i in range(1, len(encoder_dims))
            ]
        )

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps) - 1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i - 1], f_up], dim=1)
            f_down = self.convs[i - 1](f)
            feature_maps[i - 1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask


class SegModel(nn.Module):
    def __init__(self, model_depth=34):
        super().__init__()
        self.encoder = generate_model(model_depth=model_depth, n_input_channels=1)
        self.decoder = Decoder(encoder_dims=[64, 128, 256, 512], upscale=4)

    def forward(self, x):
        if x.ndim == 4:
            x = x[:, None]

        feat_maps = self.encoder(x)
        feat_maps_pooled = [torch.mean(f, dim=2) for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)
        return pred_mask


class CustomModel(nn.Module):
    def __init__(self, cfg, weight=None):
        super().__init__()
        self.cfg = cfg

        if cfg.backbone == "resnet3d":
            self.encoder = SegModel()
        else:
            self.encoder = smp.Unet(
                encoder_name=cfg.backbone,
                encoder_weights=weight,
                classes=cfg.target_size,
                activation=None,
            )
            print(self.encoder.encoder.patch_embed1.proj)
            out_channels = self.encoder.encoder.patch_embed1.proj.out_channels
            self.encoder.encoder.patch_embed1.proj = nn.Conv2d(
                cfg.in_chans, out_channels, 7, 4, 3
            )

    def forward(self, images: torch.Tensor):
        # image.shape=(b,C,H,W)
        if images.ndim == 4:
            images = images[:, None]
        images = normalization(images)
        output = self.encoder(images)
        return output


def build_model(cfg, weight="imagenet"):
    print("model_name", cfg.model_name)
    print("backbone", cfg.backbone)

    model = CustomModel(cfg, weight)
    return model


def build_model(cfg, weight="imagenet"):
    print("model_name", cfg.model_name)
    print("backbone", cfg.backbone)

    model = CustomModel(cfg, weight)

    return model
