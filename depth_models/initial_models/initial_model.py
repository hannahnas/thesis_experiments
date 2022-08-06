import torch
import torch.nn as nn
from modules.encoder import ResNetEncoder
from depth_models.baseline.decoder import ResNetDecoder


class Depth2Depth(nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.depth_encoder = ResNetEncoder(in_channels=1, type=hyper_params['encoder type'])
        self.decoder = ResNetDecoder(in_channels=256, out_channels=1)

        self._init_params()


    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch):
        masked_depth = (1 - batch['mask']) * batch['depth']

        out = self.depth_encoder(masked_depth)
        depth_pred = self.decoder(out)

        return depth_pred
        
class ChannelwiseRGB(nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.single_encoder = ResNetEncoder(in_channels=4, type=hyper_params['encoder type'])
        self.decoder = ResNetDecoder(in_channels=256, out_channels=1)

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch):
        mask = batch['mask']
        masked_rgb = (1 - mask) * batch['rgb']
        masked_depth = (1 - mask) * batch['depth']

        input = torch.cat([masked_rgb, masked_depth], dim=1)
        out = self.single_encoder(input)
        depth_pred = self.decoder(out)

        return depth_pred


class DualEncoder(nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        encoder_type = hyper_params['encoder type']
        self.rgb_encoder = ResNetEncoder(in_channels=3, type=encoder_type)
        self.depth_encoder = ResNetEncoder(in_channels=1, type=encoder_type)

        self.decoder = ResNetDecoder(in_channels=512, out_channels=1)

        self._init_params()


    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch):
        mask = batch['mask']
        masked_rgb = (1 - mask) * batch['rgb']
        masked_depth = (1 - mask) * batch['depth']

        rgb_feat = self.rgb_encoder(masked_rgb)
        depth_feat = self.depth_encoder(masked_depth)

        feat = torch.cat([rgb_feat, depth_feat], dim=1)
        depth_pred = self.decoder(feat)

        return depth_pred

