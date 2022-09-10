import torch
import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as F
from modules.encoder import ResNetEncoder
from depth_models.baseline.decoder import ResNetDecoder
from modules.evaluation import compute_eval_measures
from modules.criterions import L1_loss


class BaselineModel(pl.LightningModule):
    def __init__(self, hyper_params):
        super().__init__()

        self.hyper_params = hyper_params
        encoder_type = hyper_params['encoder type']

        self.rgb_encoder = ResNetEncoder(in_channels=3, type=encoder_type)
        self.depth_encoder = ResNetEncoder(in_channels=1, type=encoder_type)

        if 'concat edge' in hyper_params.keys():
            if hyper_params['concat edge'] == 'rgb':
                self.rgb_encoder = ResNetEncoder(in_channels=4, type=encoder_type)
                self.concat_rgb = True
                self.concat_depth = False
            if hyper_params['concat edge'] == 'depth':
                self.depth_encoder = ResNetEncoder(in_channels=2, type=encoder_type)
                self.concat_depth = True
                self.concat_rgb = False
        else:
            self.concat_rgb = False
            self.concat_depth = False

        self.decoder = ResNetDecoder(in_channels=512, out_channels=1)
        

    def forward(self, batch):
        rgb = batch['rgb']
        masked_depth = (1 - batch['mask']) * batch['depth']
        edges = batch['edges']

        if self.concat_rgb:
            rgb = torch.cat([rgb, edges], dim=1)
            print('hello rgb')
        if self.concat_depth:
            masked_depth = torch.cat([masked_depth, edges], dim=1)
            print('hello depth')

        rgb_feat = self.rgb_encoder(rgb)
        depth_feat = self.depth_encoder(masked_depth)

        feat = torch.cat([rgb_feat, depth_feat], dim=1)
        pred = self.decoder(feat)

        return pred
    
    def _get_loss(self, batch):
        depth_gt = batch['depth']
        mask = batch['mask']

        depth_pred = self.forward(batch)
        l1_loss = L1_loss(depth_pred, depth_gt, mask)

        return l1_loss
   
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hyper_params['lr'])

        return {"optimizer": optimizer, "monitor": self.hyper_params['monitor']}

    def training_step(self, batch, batch_idx):
        l1_loss = self._get_loss(batch)
        self.log('train_l1_depth_loss', l1_loss)

        return l1_loss

    def validation_step(self, batch, batch_idx):
        l1_loss = self._get_loss(batch)
        self.log('val_l1_depth_loss', l1_loss)


    def test_step(self, batch, batch_idx):
        l1_loss = self._get_loss(batch)
        self.log('test_l1_depth_loss', l1_loss)

        depth_pred = self.forward(batch)
        depth_gt = batch['depth']
        mask = batch['mask']
        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_eval_measures(depth_gt, depth_pred, mask)

        self.log('abs rel', abs_rel)
        self.log('sq rel', sq_rel)
        self.log('rmse', rmse)
        self.log('rmse log', rmse_log)
        self.log('delta 1.25', a1)
        self.log('delta 1.25^2', a2)
        self.log('delta 1.25^3', a3)
