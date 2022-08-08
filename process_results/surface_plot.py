import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
import pytorch_lightning as pl
from depth_models.skip_attention.pl_skipattentionmodel import SkipAttentionModel
from depth_models.baseline.pl_baseline import BaselineModel
from depth_models.skipnet.pl_skipnet import SkipNetModel
from depth_models.edge_attention.pl_edgeattentionmodel import EdgeAttentionModel


def get_geometry(color, depth, mask):
    color = color * 255
    mask = mask.astype(bool)
    color[mask, 2] += 150
    color = o3d.geometry.Image(np.ascontiguousarray(color).astype(np.uint8))

    depth = o3d.geometry.Image(np.ascontiguousarray(depth).astype(np.float32))
    cam = o3d.camera.PinholeCameraIntrinsic()
    
    cam.set_intrinsics(
        width=248,
        height=248,
        fx=147.22,
        fy=147.80,
        cx=124.5,
        cy=124.5
    )

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic=cam)

    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0],
                   [0, 0, -1, 0], [0, 0, 0, 1]])

    # o3d.visualization.draw_geometries([pcd])
    return pcd

def generate_mask(img_size):
    """
            Create mask with box in random location.
            Create another slightly bigger mask in the same location.
    """
    H, W = img_size, img_size
    mask = torch.zeros((H, W))
    box_size = round(H * 0.3)

    x_loc = np.random.randint(0, W - box_size)
    y_loc = np.random.randint(0, H - box_size)

    mask[y_loc:y_loc+box_size, x_loc:x_loc+box_size] = 1

    return mask

def plot_2D():
    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(rgb)
    ax[1].imshow(depth, cmap='viridis')
    ax[2].imshow(edge, cmap='gray')
    ax[3].imshow(mask, cmap='gray')
    plt.tight_layout()

def plot_3D(gt, pred):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Input + GT', width=540, height=540, left=0, top=0)
    vis.add_geometry(gt)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='Predicted Depth', width=540, height=540, left=960, top=0)
    vis2.add_geometry(pred)

    while True:
        vis.update_geometry(gt)
        if not vis.poll_events():
            break
        vis.update_renderer()

        vis2.update_geometry(pred)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

    vis.destroy_window()
    vis2.destroy_window()

model2results_path = {
    'SkipNet': '/home/hannah/Documents/Thesis/thesis_experiments/results/SkipNet_batch8_lr0.0001_10.pickle',
    'SkipAttention': '/home/hannah/Documents/Thesis/thesis_experiments/results/GatedSkipAttention_batch8_lr0.0001_10.pickle',
    'MultiScaleSkipAttention': '/home/hannah/Documents/Thesis/thesis_experiments/results/MultiScaleSkipAttention_batch8_lr0.0001_10.pickle',
    'MultiScaleSkipAttentionSmooth': '/home/hannah/Documents/Thesis/thesis_experiments/results/MultiScaleSkipAttentionWithSmoothness_batch8_lr0.0001_10.pickle',
    'EdgeAttention': '/home/hannah/Documents/Thesis/thesis_experiments/results/EdgeAttentionModel_batch8_lr0.0001_10.pickle'
}

modelname2class = {
    'BaselineModel': BaselineModel,
    'SkipNetModel': SkipNetModel,
    'EdgeAttentionModel': EdgeAttentionModel,
    'SkipAttentionModel': SkipAttentionModel,   
}

if __name__ == "__main__":
    # Choose a model from:
    # evaluate = 'SkipNet'
    # evaluate = 'EdgeAttention'
    evaluate = 'SkipAttention'
    # evaluate = 'MultiScaleSkipAttention'
    # evaluate = 'MultiScaleSkipAttentionSmooth'

    # Choose image: 0, 1, 2, 3
    # index = 2 # flat surface
    # index = 1 # complicated scene
    # index = 0
    index = 0

    results_path = model2results_path[evaluate]
    file = open(results_path, 'rb')
    results = pickle.load(file)
    print(results['model_path'])
    model_class = modelname2class[results['hyper_params']['model class']]

    model = model_class.load_from_checkpoint(results['model_path'], hyper_params=results['hyper_params'])

    samples = ['032988', '040878', '106194', '066096']
    sample_path = '/home/hannah/Documents/Thesis/val_sample/'
    
    # mask = generate_mask(248)
    # np.save(f'{sample_path}0_mask.npy', mask)

    rgb = np.load(f'{sample_path}{samples[index]}_image.npy')[2:250, 2:250, :3]
    depth = np.load(f'{sample_path}{samples[index]}_depth.npy')[2:250, 2:250]
    edge = np.load(f'{sample_path}{samples[index]}_gray_edges.npy')[2:250, 2:250]
    mask = np.load(f'{sample_path}0_mask.npy')

    batch = {
        'rgb': torch.Tensor(rgb).permute(2, 0, 1).unsqueeze(0),
        'depth': torch.Tensor(depth).unsqueeze(0).unsqueeze(0),
        'mask': torch.Tensor(mask).unsqueeze(0).unsqueeze(0),
        'edges': torch.Tensor(edge).unsqueeze(0).unsqueeze(0)
    }

    depth_pred = model(batch)

    if results['hyper_params']['model name'] == 'EdgeAttentionModel':
        depth_pred, _ = depth_pred
    if 'multiscale' in results['hyper_params'].keys() and results['hyper_params']['multiscale']:
        _, _, depth_pred = depth_pred

    completed_depth = batch['depth'] * (1 - batch['mask']) + depth_pred * batch['mask']
    completed_depth = completed_depth.cpu().detach().numpy()[0, 0]

    gt = get_geometry(rgb, depth, mask)
    pred = get_geometry(rgb, completed_depth, mask)
    plot_3D(gt, pred)
