import torch
import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as F
from depth_models.skipnet.skipnet import SkipNet, SingleEncoderSkipNet
from depth_models.skipnet.skipnet_rgb_with_edge import SkipNetRGBwithEdge
from modules.evaluation import compute_eval_measures
from modules.criterions import L1_loss, single_disp_smoothness


class SkipNetModel(pl.LightningModule):
    def __init__(self, hyper_params):
        super().__init__()

        self.hyper_params = hyper_params
        self.multiscale = hyper_params['multiscale']
        self.smoothness_loss = hyper_params['smoothness loss']
        if hyper_params['concat rgb']:
            self.model = SkipNetRGBwithEdge(hyper_params)
        if hyper_params['single encoder']:
            self.model = SingleEncoderSkipNet(hyper_params)
        else:
            self.model = SkipNet(hyper_params)

        self.use_edges = hyper_params['use edges']


    def forward(self, batch):

        pred = self.model(batch)

        return pred
    
    def _get_loss(self, batch):
        depth_gt = batch['depth']
        mask = batch['mask']

        depth_pred = self.forward(batch)
        
        if self.multiscale:
            _, _, _, size = depth_gt.shape
            out_down2x, out_down1x, out_full = depth_pred

            gt_down2x = F.interpolate(depth_gt, size=(size//4, size//4), mode='bilinear', align_corners=True)
            mask_down2x = F.interpolate(depth_gt, size=(size//4, size//4), mode='nearest')
            
            gt_down1x = F.interpolate(depth_gt, size=(size//2, size//2), mode='bilinear', align_corners=True)
            mask_down1x = F.interpolate(depth_gt, size=(size//2, size//2), mode='nearest')

            l1_down2x = L1_loss(out_down2x, gt_down2x, mask_down2x)
            l1_down1x = L1_loss(out_down1x, gt_down1x, mask_down1x)
            l1_full = L1_loss(out_full, depth_gt, mask)
            
            l1_loss = (l1_down2x, l1_down1x, l1_full)
        else:
            l1_loss = L1_loss(depth_pred, depth_gt, mask)

        if self.smoothness_loss:
            if self.multiscale:
                depth_pred = out_full
            smooth_loss = single_disp_smoothness(batch['rgb'], depth_pred)
        else:
            smooth_loss = None

        return l1_loss, smooth_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hyper_params['lr'])

        return {"optimizer": optimizer, "monitor": self.hyper_params['monitor']}

    def training_step(self, batch, batch_idx):
        l1_loss, smooth_loss = self._get_loss(batch)
        
        if self.multiscale:
            (l1_down2x, l1_down1x, l1_full) = l1_loss
            self.log('train_l1_downsampled2x', l1_down2x)
            self.log('train_l1_downsampled1x', l1_down1x)
            self.log('train_l1_full', l1_full)
            l1_loss = l1_down2x + l1_down1x + l1_full
        else:
            self.log('train_l1_depth_loss', l1_loss)
        
        if self.smoothness_loss:
            self.log('train_smoothness_loss', smooth_loss)
            loss = self.hyper_params['lambda l1'] * l1_loss - self.hyper_params['lambda smooth'] * smooth_loss
            self.log('train_weighted_loss', loss)
        else:
            loss = l1_loss
        
        return loss

    def validation_step(self, batch, batch_idx):
        l1_loss, smooth_loss = self._get_loss(batch)
        
        if self.multiscale:
            (l1_down2x, l1_down1x, l1_full) = l1_loss
            self.log('val_l1_downsampled2x', l1_down2x)
            self.log('val_l1_downsampled1x', l1_down1x)
            self.log('val_l1_full', l1_full)
            l1_loss = (l1_down2x + l1_down1x + l1_full) / 3
        else:
            self.log('val_l1_depth_loss', l1_loss)
        
        if self.smoothness_loss:
            self.log('val_smoothness_loss', smooth_loss)
            loss = self.hyper_params['lambda l1'] * l1_loss - self.hyper_params['lambda smooth'] * smooth_loss
        else:
            loss = l1_loss

        self.log('val_loss_total', loss)


    def test_step(self, batch, batch_idx):
        # Loss terms
        l1_loss, smooth_loss = self._get_loss(batch)
        if self.multiscale:
            (l1_down2x, l1_down1x, l1_full) = l1_loss
            self.log('test_l1_downsampled2x', l1_down2x)
            self.log('test_l1_downsampled1x', l1_down1x)
            self.log('test_l1_full', l1_full)
            l1_loss = l1_down2x + l1_down1x + l1_full
        else:
            self.log('test_l1_depth_loss', l1_loss)
        
        if self.smoothness_loss:
            self.log('test_smoothness_loss', smooth_loss)
            weighted_loss = self.hyper_params['lambda l1'] * l1_loss + self.hyper_params['lambda smooth'] * smooth_loss
            self.log('test_weighted_loss', weighted_loss)

        # Evaluation measures
        depth_pred = self.forward(batch)
        if self.multiscale:
            _, _, depth_pred = depth_pred
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
