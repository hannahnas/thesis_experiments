# from modules import EdgeAttentionModule, ResNetBlockDec
import torch
import torch.nn as nn
from modules.blocks import ResNetBlockDec
from modules.encoder import ResNetEncoder

class EdgeAttention(nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self._create_network()
        self._init_params()

    def  _create_network(self):
        self.rgb_encoder = ResNetEncoder(in_channels=3, type=self.hyper_params['encoder type'])
        self.depth_encoder = ResNetEncoder(in_channels=1, type=self.hyper_params['encoder type'])

        self.to_hidden = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.attn1 = EdgeAttentionModule(128, 64)
        self.attn2 = EdgeAttentionModule(64, 32)
        self.attn3 = EdgeAttentionModule(32, 16)

        self.conv_feat = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.conv_edge = nn.Conv2d(16, 1, kernel_size=3, padding=1)
    

    def forward(self, rgb, depth):
        rgb_feat = self.rgb_encoder(rgb)
        depth_feat = self.depth_encoder(depth)
        
        feat = torch.cat([rgb_feat, depth_feat], dim=1)

        feat = self.to_hidden(feat)
        feat, edges = self.attn1(feat, feat)
        feat, edges = self.attn2(feat, edges)
        feat, edges = self.attn3(feat, edges)


        feat = self.conv_feat(feat)
        edges = self.conv_edge(edges)

        return feat, edges

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class EdgeAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.dec_block = nn.Sequential(
            ResNetBlockDec(in_channels, out_channels, subsample=True),
            ResNetBlockDec(out_channels, out_channels, subsample=False)
        )

        self.edge_block = nn.Sequential(
            ResNetBlockDec(in_channels, out_channels, subsample=True),
            ResNetBlockDec(out_channels, out_channels, subsample=False)
        )
        

    def forward(self, feat, edges):

        feat = self.dec_block(feat)
        edges = self.edge_block(edges)

        attention = torch.sigmoid(edges)

        feat = attention * feat + feat

        return feat, edges
