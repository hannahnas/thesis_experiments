from tabnanny import check
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import numpy as np
import pickle
import pytorch_lightning as pl
from depth_models.skip_attention.pl_skipattentionmodel import SkipAttentionModel
from depth_models.baseline.pl_baseline import BaselineModel
from depth_models.skipnet.pl_skipnet import SkipNetModel
from depth_models.edge_attention.pl_edgeattentionmodel import EdgeAttentionModel
import matplotlib.pyplot as plt

from depth_models.skipnet.skipnet import SingleEncoderSkipNet

class ModelOutputWrapper(torch.nn.Module):
    def __init__(self, model): 
        super(ModelOutputWrapper, self).__init__()
        self.model = model
        
    def forward(self, input_tensor):
        rgb = input_tensor[:,:3]
        depth = input_tensor[:, 3].unsqueeze(1)
        mask = input_tensor[:, 4].unsqueeze(1)
        edges = input_tensor[:, 5].unsqueeze(1)

        batch = {
            'rgb': rgb,
            'depth': depth,
            'mask': mask,
            'edges': edges
        }

        return self.model(batch)

class MaskPredictionTarget:
    def __init__(self, mask):
        self.mask = mask

    def __call__(self, model_output):
        return (model_output * self.mask).sum()

def get_cam_img(model, target_layer, input_tensor, show_on):

    cam = GradCAM(model=model, target_layers=target_layer, use_cuda=False)
    targets = [MaskPredictionTarget(mask)]

    with cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        cam_image = show_cam_on_image(show_on, grayscale_cam, use_rgb=True)
    
    return cam_image


# Input
samples = ['032988', '040878', '106194', '066096']
sample_path = '/home/hannah/Documents/Thesis/val_sample/'
index = 1

rgb_np = np.load(f'{sample_path}{samples[index]}_image.npy')[2:250, 2:250, :3]
depth_np = np.load(f'{sample_path}{samples[index]}_depth.npy')[2:250, 2:250]

depth_3channel = np.repeat(depth_np[:, :, np.newaxis], 3, axis=2)
print(depth_3channel.shape)
edge_np = np.load(f'{sample_path}{samples[index]}_gray_edges.npy')[2:250, 2:250]
mask_np = np.load(f'{sample_path}0_mask.npy')

rgb = torch.Tensor(rgb_np).permute(2, 0, 1).unsqueeze(0)
depth = torch.Tensor(depth_np).unsqueeze(0).unsqueeze(0)
mask = torch.Tensor(mask_np).unsqueeze(0).unsqueeze(0)
edges = torch.Tensor(edge_np).unsqueeze(0).unsqueeze(0)
input_tensor = torch.cat([rgb, depth, mask, edges], dim=1)

model2results_path = {
    'SingleEncoderSkipnet': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SingleEncoderSkipNet_batch8_20.pickle',
    'DualEncoderSkipnet': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNet_batch8_20.pickle',
    'SingleEncoderSkipnetEdge': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SingleEncoderSkipNetWithEdge_batch8_20.pickle',
    'DualEncoderSkipnetEdge': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithEdges_batch8_20.pickle',
    # 'SkipAttention': '',
    # 'MultiScaleSkipAttention': '',
    # 'MultiScaleSkipAttentionSmooth': '',
    'EdgeAttention': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/EdgeAttentionModel_batch8_20.pickle'
}

# Model
modelname2class = {
    'BaselineModel': BaselineModel,
    'SkipNetModel': SkipNetModel,
    'EdgeAttentionModel': EdgeAttentionModel,
    'SkipAttentionModel': SkipAttentionModel,   
}

evaluate = 'DualEncoderSkipnet'
results_path = model2results_path[evaluate]
file = open(results_path, 'rb')
results = pickle.load(file)
model_path = results['model_path']
params = results['hyper_params']
if 'multiscale' not in params.keys():
    params['multiscale'] = False
if 'smoothness loss' not in params.keys():
    params['smoothness loss'] = False


model_class = modelname2class[results['hyper_params']['model class']]

pl_model = model_class.load_from_checkpoint(results['model_path'], hyper_params=params)

with torch.no_grad():
    batch = {
            'rgb': rgb,
            'depth': depth,
            'mask': mask,
            'edges': edges
        }
    depth_pred = pl_model(batch)#[0].permute(1, 2, 0)

    # depth_pred = mask[0] * depth_pred + (1-mask[0]) * depth[0]

model = ModelOutputWrapper(pl_model.model)

if params['single encoder']:
    fig, ax = plt.subplots(4, 4)
    ax[0, 0].set_title('input')
    ax[0, 0].imshow(rgb_np)
    ax[0, 0].axis('off')

    ax[0, 1].imshow((1 - mask_np) * depth_np, cmap='viridis')
    ax[0, 1].set_title('input')
    ax[0, 1].axis('off')

    # completed_depth = batch['depth'] * (1 - batch['mask']) + depth_pred * batch['mask']

    # completed_depth = completed_depth.cpu().numpy()[0, 0]
    # ax[0, 2].imshow(completed_depth, cmap='viridis')
    # ax[0, 2].set_title('prediction')
    ax[0, 2].axis('off')

    ax[0, 3].imshow(edge_np, cmap='gray')
    ax[0, 3].axis('off')

    enc_0 = [model.model.single_encoder.layer_0[-1]]
    enc_1 = [model.model.single_encoder.layer_1[1].act_fn]
    enc_2 = [model.model.single_encoder.layer_2[1].act_fn]
    enc_3 = [model.model.single_encoder.layer_3[1].act_fn]
    encoder_layers = [enc_0, enc_1, enc_2, enc_3]
    for i, layer in enumerate(encoder_layers):
        cam_img = get_cam_img(model, layer, input_tensor, rgb_np)

        ax[1, i].set_title(f'Encoder activation map {i}')
        ax[1, i].imshow(cam_img)
        ax[1, i].axis('off')

    enc_gate0 = [model.model.single_encoder.layer_0[0].sigmoid]
    enc_gate1 = [model.model.single_encoder.layer_1[1].net[0].sigmoid]
    enc_gate2 = [model.model.single_encoder.layer_2[1].net[0].sigmoid]
    enc_gate3 = [model.model.single_encoder.layer_3[1].net[0].sigmoid]
    gate_layers = [enc_gate0, enc_gate1, enc_gate2, enc_gate3]
    for i, layer in enumerate(gate_layers):
        cam_img = get_cam_img(model, layer, input_tensor, rgb_np)
        ax[2, i].set_title(f'Encoder gating map {i}')
        ax[2, i].imshow(cam_img)
        ax[2, i].axis('off')

    red_feat = [model.model.reduce_features[-1]]
    dec_3 = [model.model.dec_layer_3[1].act_fn]
    dec_2 = [model.model.dec_layer_2[1].act_fn]
    dec_1 = [model.model.dec_layer_1[1].act_fn]
    decoder_layers = [red_feat, dec_3, dec_2, dec_3]

    for i, layer in enumerate(decoder_layers):
        cam_img = get_cam_img(model, layer, input_tensor, rgb_np)
        ax[3, i].set_title(f'Decoder activation map {i}')
        ax[3, i].imshow(cam_img)
        ax[3, i].axis('off')
    # print(model.model.single_encoder.layer_1[1].act_fn)
    
    

if params['single encoder'] == False:
    depth_visual = depth_3channel / depth_3channel.max()
    fig, ax = plt.subplots(6, 4)

    ax[0, 0].set_title('input')
    ax[0, 0].imshow(rgb_np)
    ax[0, 0].axis('off')

    ax[0, 1].imshow((1 - mask_np) * depth_np, cmap='viridis')
    ax[0, 1].set_title('input')
    ax[0, 1].axis('off')

    completed_depth = batch['depth'] * (1 - batch['mask']) + depth_pred * batch['mask']
    completed_depth = completed_depth.cpu().numpy()[0, 0]
    ax[0, 2].imshow(completed_depth, cmap='viridis')
    ax[0, 2].set_title('prediction')
    ax[0, 2].axis('off')
    ax[0, 3].imshow(edge_np, cmap='gray')
    ax[0, 3].axis('off')

    rgb_enc_0 = [model.model.rgb_encoder.layer_0[-1]]
    rgb_enc_1 = [model.model.rgb_encoder.layer_1[1].act_fn]
    rgb_enc_2 = [model.model.rgb_encoder.layer_2[1].act_fn]
    rgb_enc_3 = [model.model.rgb_encoder.layer_3[1].act_fn]
    rgb_encoder_layers = [rgb_enc_0, rgb_enc_1, rgb_enc_2, rgb_enc_3]
    for i, layer in enumerate(rgb_encoder_layers):
        cam_img = get_cam_img(model, layer, input_tensor, rgb_np)

        ax[1, i].set_title(f'RGB encoder activation {i}')
        ax[1, i].imshow(cam_img)
        ax[1, i].axis('off')

    rgb_enc_gate0 = [model.model.depth_encoder.layer_0[0].sigmoid]
    rgb_enc_gate1 = [model.model.depth_encoder.layer_1[1].net[0].sigmoid]
    rgb_enc_gate2 = [model.model.depth_encoder.layer_2[1].net[0].sigmoid]
    rgb_enc_gate3 = [model.model.depth_encoder.layer_3[1].net[0].sigmoid]
    rgb_gate_layers = [rgb_enc_gate0, rgb_enc_gate1, rgb_enc_gate2, rgb_enc_gate3]
    for i, layer in enumerate(rgb_gate_layers):
        cam_img = get_cam_img(model, layer, input_tensor, rgb_np)

        ax[2, i].set_title(f'RGB encoder gate {i}')
        ax[2, i].imshow(cam_img)
        ax[2, i].axis('off')

    depth_enc_0 = [model.model.depth_encoder.layer_0[-1]]
    depth_enc_1 = [model.model.depth_encoder.layer_1[1].act_fn]
    depth_enc_2 = [model.model.depth_encoder.layer_2[1].act_fn]
    depth_enc_3 = [model.model.depth_encoder.layer_3[1].act_fn]
    depth_encoder_layers = [depth_enc_0, depth_enc_1, depth_enc_2, depth_enc_3]
    
    for i, layer in enumerate(depth_encoder_layers):
        cam_img = get_cam_img(model, layer, input_tensor, depth_visual)

        ax[3, i].set_title(f'Depth encoder activation {i}')
        ax[3, i].imshow(cam_img)
        ax[3, i].axis('off')

    depth_enc_gate0 = [model.model.depth_encoder.layer_0[0].sigmoid]
    depth_enc_gate1 = [model.model.depth_encoder.layer_1[1].net[0].sigmoid]
    depth_enc_gate2 = [model.model.depth_encoder.layer_2[1].net[0].sigmoid]
    depth_enc_gate3 = [model.model.depth_encoder.layer_3[1].net[0].sigmoid]
    depth_gate_layers = [depth_enc_gate0, depth_enc_gate1, depth_enc_gate2, depth_enc_gate3]
    
    for i, layer in enumerate(depth_gate_layers):
        cam_img = get_cam_img(model, layer, input_tensor, depth_visual)

        ax[4, i].set_title(f'Depth encoder gate {i}')
        ax[4, i].imshow(cam_img)
        ax[4, i].axis('off')

    red_feat = [model.model.reduce_features[-1]]
    dec_3 = [model.model.dec_layer_3[1].act_fn]
    dec_2 = [model.model.dec_layer_2[1].act_fn]
    dec_1 = [model.model.dec_layer_1[1].act_fn]
    decoder_layers = [red_feat, dec_3, dec_2, dec_3]
    for i, layer in enumerate(decoder_layers):
        cam_img = get_cam_img(model, layer, input_tensor, rgb_np)
        ax[5, i].set_title(f'Decoder activation {i}')
        ax[5, i].imshow(cam_img)
        ax[5, i].axis('off')


plt.tight_layout()
plt.show()
# plt.savefig('./grad_cam_images/dual_enc_edge.png')
