import torch
import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as F
from modules.encoder import ResNetEncoder
from depth_models.edge_attention.edge_attention import EdgeAttention
from modules.evaluation import compute_eval_measures
from modules.criterions import L1_loss, cross_entropy_loss2d

class EdgeAttentionModel(pl.LightningModule):
    def __init__(self, hyper_params):
        super().__init__()

        self.hyper_params = hyper_params
        self.model = EdgeAttention(hyper_params)


    def forward(self, batch):
        rgb = batch['rgb']
        depth = batch['depth']
        mask = batch['mask']
        masked_depth = (1 - mask) * depth 

        depth, edges = self.model(rgb, masked_depth)

        return depth, edges
    
    def _get_losses(self, batch):
        depth_gt = batch['depth']
        edges_gt = batch['edges']
        # edges_gt = torch.cat([batch['edges'].logical_not(), batch['edges']], dim=1).long()
        mask = batch['mask']

        depth_pred, edges_pred = self.forward(batch)

        l1_loss = L1_loss(depth_pred, depth_gt, mask)
        bce_loss = cross_entropy_loss2d(edges_pred, edges_gt)

        return l1_loss, bce_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hyper_params['lr'])

        return {"optimizer": optimizer, "monitor": self.hyper_params['monitor']}

    def training_step(self, batch, batch_idx):
        l1_loss, bce_loss = self._get_losses(batch)
        total_loss = self.hyper_params['lambda depth'] * l1_loss + self.hyper_params['lambda edge'] * bce_loss
        self.log('train_l1_depth_loss', l1_loss)
        self.log('train_bce_edge_loss', bce_loss)
        self.log('total_train_loss', total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        l1_loss, bce_loss = self._get_losses(batch)
        total_loss = self.hyper_params['lambda depth'] * l1_loss + self.hyper_params['lambda edge'] * bce_loss
        self.log('val_l1_depth_loss', l1_loss)
        self.log('val_bce_edge_loss', bce_loss)
        self.log('total_val_loss', total_loss)


    def test_step(self, batch, batch_idx):
        l1_loss, bce_loss = self._get_losses(batch)
        self.log('test_l1_depth_loss', l1_loss)
        self.log('test_bce_edge_loss', bce_loss)

        depth_pred, _ = self.forward(batch)
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
