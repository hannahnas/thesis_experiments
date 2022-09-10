from re import I
import os
import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
import pytorch_lightning as pl
from depth_models.initial_models.pl_initial_models import InitialModel
from depth_models.skip_attention.pl_skipattentionmodel import SkipAttentionModel
from depth_models.baseline.pl_baseline import BaselineModel
from depth_models.skipnet.pl_skipnet import SkipNetModel
from depth_models.edge_attention.pl_edgeattentionmodel import EdgeAttentionModel

VAL_samples = ('055671', '010486', '152419', '093810', '086143',  '036993', '115594', '108930', '096545', '013078', '087940', '060998', '019780', '139194', '100616', '100641', '015534', '064985', '136682', '044201', '114333', '141304', '007006', '022580', '080888', '036947', '038223', '097128', '149613', '144047')
TEST = ('066240', '030875', '112963', '027774', '015840', '015757', '075155', '050489', '005184', '047119', '116784', '119945', '015822', '037038', '149723', '086213', '099460', '050484', '145244', '142731', '055168', '085349', '017092', '023961', '077285', '085366', '128823', '015828', '109100', '092406')


def parse_camera_info(height=720, width=1280):
    """ extract intrinsic and extrinsic matrix
    """
    camera_info = np.loadtxt('/home/hannah/Documents/Thesis/data/Structured3D/scene_00078/2D_rendering/1448/perspective/full/2/camera_pose.txt')
    xfov = camera_info[9]
    yfov = camera_info[10]

    K = np.diag([1, 1, 1])

    K[0, 2] = width / 2
    K[1, 2] = height / 2

    fx = K[0, 2] / np.tan(xfov)
    fy = K[1, 2] / np.tan(yfov)
    print(fx, fy)

    return fx, fy


# def save_viewpoints(pcd):
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     vis.add_geometry(pcd)
#     vis.run()
    
#     for i in range(5):
#         ctr = vis.get_view_control()
#         ctr.rotate(i*72, 0.0)
#         vis.update_geometry()
#         vis.poll_events()
#         vis.update_renderer()
#         # image = vis.capture_screen_float_buffer(False)
#         vis.capture_screen_image('test.png', True)
#         # plt.imsave(f'./plots/view_{i}.png',
#         #                     np.asarray(image),
#         #                     dpi=1)
    
#     vis.destroy_window()
#     print('done')

# def custom_draw_geometry_with_custom_fov(pcd, fov_step):
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     vis.add_geometry(pcd, material=open3d.cpu.pybind.visualization.rendering.MaterialRecord)
#     ctr = vis.get_view_control()
#     print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
#     ctr.change_field_of_view(step=fov_step)
#     print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
#     vis.run()
#     vis.destroy_window()


def get_geometry(color, depth, mask):
    color = color * 255
    mask = mask.astype(bool)
    color[mask, 2] += 150
    # color[mask, 2] = 200
    # color[mask, 0] = 0
    # color[mask, 1] = 150
    color = o3d.geometry.Image(np.ascontiguousarray(color).astype(np.uint8))

    depth = o3d.geometry.Image(np.ascontiguousarray(depth).astype(np.float32))
    cam = o3d.camera.PinholeCameraIntrinsic()
    fovx, fovy = parse_camera_info()
    cam.set_intrinsics(
        width=248,
        height=248,
        # fx = 147.22,
        fx=fovx/2.8125,
        # fy = 147.80,
        fy=fovy/2.8125, 
        cx=124.5-4,
        cy=124.5-4
    )
    

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic=cam)

    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0],
                   [0, 0, -1, 0], [0, 0, 0, 1]])

    # o3d.visualization.draw_geometries([pcd])
    return pcd

def capture_image(vis, i):
    image = vis.capture_screen_float_buffer()
    image = np.asarray(image)
    # image = crop_image_v2(image)
    print(VIEWS_DIR)
    plt.imsave(f'{VIEWS_DIR}/view_{i}.png',
                        image,
                        dpi=1)
    
    return False

def capture_image_GT(vis, i):
    image = vis.capture_screen_float_buffer()
    image = np.asarray(image)
    # image = crop_image_v2(image)
    plt.imsave(f'{GT_DIR}/view_{i}.png',
                        image,
                        dpi=1)
    
    return False

def custom_draw_geometry_with_rotation(pcd):
    global i
    i = 0

    def rotate_view(vis):
        global i
        ctr = vis.get_view_control()
        ctr.rotate(400.0, 0.0)
        if i < 5:
            capture_image(vis, i)
            i += 1

        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd], rotate_view, width=510, height=510)


def custom_draw_geometry_with_key_callback(pcd):

    def capture_screen(vis):
        global i
        image = vis.capture_screen_float_buffer()
        image = np.asarray(image)
        plt.imsave(f'{VIEWS_DIR}/view_{i}.png',
                        image,
                        dpi=1)
        i += 1
        return False

    
    def rotate_left(vis):
        ctr = vis.get_view_control()
        ctr.rotate(350.0, 0.0)

        return False

    def rotate_right(vis):
        ctr = vis.get_view_control()
        ctr.rotate(-350.0, 0.0)

        return False

    def rotate_up(vis):
        ctr = vis.get_view_control()
        ctr.rotate(0.0, 560.0)

        return False

    def rotate_down(vis):
        ctr = vis.get_view_control()
        ctr.rotate(0.0, -560.0)

        return False


    key_to_callback = {}
    key_to_callback[ord("R")] = rotate_right
    key_to_callback[ord("L")] = rotate_left
    key_to_callback[ord("U")] = rotate_up
    key_to_callback[ord("D")] = rotate_down
    key_to_callback[ord("S")] = capture_screen
    global i
    i = 0
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback, width=510, height=510)


def save_GT_pointcloud_rotation(pcd):
    global i
    i = 0

    def rotate_view(vis):
        global i
        ctr = vis.get_view_control()
        ctr.rotate(400.0, 0.0)
        if i < 5:
            capture_image_GT(vis, i)
            i += 1

        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd], rotate_view, width=510, height=510)

# def get_geometry_views(pcd):
#     left_pcd = pcd.rotate(rotation=[45, 0, 0], type=o3d.geometry.RotationType.AxisAngle)
#     right_pcd = pcd.rotate(rotation=[-45, 0, 0], type=open3d.geometry.RotationType.AxisAngle)
#     up_pcd = pcd.rotate(rotation=[0, 90, 0], type=o3d.geometry.RotationType.AxisAngle)

#     return left_pcd, pcd, right_pcd, up_pcd

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

def concat_views(path, evaluate):
    fig, ax = plt.subplots(1, 4)
    for i in range(4):
        img = plt.imread(f'{path}/view_{i}.png')
        ax[i].imshow(img)
        ax[i].axis('off')

    # plt.tight_layout()
    plt.savefig(f'{path}/{evaluate}_views.pdf', format='pdf', bbox_inches='tight', dpi=1200)

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
    # Modality experiment
    'depth2depth': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment0/Depth2DeptInitialModel_batch8_29.pickle',
    'earlyfus': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment0/ChannelWiseInitialModel_batch8_29.pickle',
    'middle_fuse_maskRGB': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment0/DualEncoderInitialModel_batch8_29.pickle',
    'middlefusion': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment0/DualEncoderInitialModel_batch8_29.pickle',

    # Conv experiment
    'RegularConv': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/Baseline_batch8_29.pickle',
    'DeformableConv': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/DeformableBaseline10epochs_batch8_49.pickle',
    'GatedConv': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedBaseline_batch8_29.pickle',
    'GatedDeformConv': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedDeformableBaseline_batch8_49.pickle',
    # Skip connections
    'SingleEncoderSkipnet': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SingleEncoderSkipNet_batch8_20.pickle',
    'DualEncoderSkipnet': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNet_batch8_20.pickle',
    # Edge channel
    'SingleEncoderSkipnetEdge': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SingleEncoderSkipNetWithEdge_batch8_20.pickle',
    'DualEncoderSkipnetEdge': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithEdges_batch8_20.pickle',
    'DualEncoderSkipnetRGBwithEdge': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithRGBEdges_batch8_29.pickle',
    # Edge attention
    'SkipAttention': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/GatedSkipAttention_batch8_30.pickle',
    'EdgeAttention': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/EdgeAttentionModel_batch8_20.pickle',
    'EdgeAttentionV2': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/EdgeAttentionModelPretrainBCE_V2_batch8_79.pickle',
    # Loss functions
    'SkipAttentionSmooth': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/SkipAttention+MeanSmoothness_batch8_79.pickle',
    'SkipAttentionMultiscale': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScale1x1SkipAttention_batch8_109.pickle',
    'SkipNetEdgeSmooth': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdge+MeanSmoothness_batch8_79.pickle',
    'SkipNetEdgeMultiscale': '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWith1x1MultiscaleSmoothness_batch8_109.pickle',
}

modelname2class = {
    'BaselineModel': BaselineModel,
    'SkipNetModel': SkipNetModel,
    'EdgeAttentionModel': EdgeAttentionModel,
    'SkipAttentionModel': SkipAttentionModel,
    'InitialModel': InitialModel   
}

if __name__ == "__main__":
    split = 'val'
    # modality experiment
    evaluate = 'depth2depth'
    # evaluate = 'earlyfus'
    # evaluate = 'middlefusion'


    # Gated conv
    # evaluate = 'RegularConv'
    # evaluate = 'GatedConv'
    # evaluate = 'DeformableConv'
    # evaluate = 'GatedDeformConv'

    # Choose a model from:
    # evaluate = 'SingleEncoderSkipnet'
    # evaluate = 'DualEncoderSkipnet'
    # evaluate = 'SingleEncoderSkipnetEdge'
    # evaluate = 'DualEncoderSkipnetEdge'
    # evaluate = 'DualEncoderSkipnetRGBwithEdge'

    # evaluate = 'EdgeAttention'
    # evaluate = 'EdgeAttentionV2'
    # evaluate = 'SkipAttention'

    # evaluate = 'SkipAttentionSmooth'
    # evaluate = 'SkipAttentionMultiscale'
    # evaluate = 'SkipNetEdgeSmooth'
    # evaluate = 'SkipNetEdgeMultiscale'

    # Choose image: 0, 1, 2, 3, 4
    # index = 2 # flat surface
    # index = 1 # complicated scene
    # index = 0
    index = 18

    results_path = model2results_path[evaluate]
    file = open(results_path, 'rb')
    results = pickle.load(file)
    print(results['model_path'])
    model_class = modelname2class[results['hyper_params']['model class']]

    params = results['hyper_params']
    if 'multiscale' not in params.keys():
        params['multiscale'] = False
    if 'smoothness loss' not in params.keys():
        params['smoothness loss'] = False
    if 'concat rgb' not in params.keys():
        params['concat rgb'] = False

    model = model_class.load_from_checkpoint(results['model_path'], hyper_params=params)

    samples = ['032988', '040878', '106194', '066096', '000462']
    samples_path = '/home/hannah/Documents/Thesis/val_sample/'
    
    # mask = generate_mask(248)
    # np.save(f'{samples_path}5_mask.npy', mask)

    model_dir = f"plots/{evaluate}"
    sample_dir = f"plots/{evaluate}/{split}_sample_{VAL_samples[index]}"
    parent_dir = '/home/hannah/Documents/Thesis/thesis_experiments'
    # parent_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(parent_dir, model_dir)
    sample_path = os.path.join(parent_dir, sample_dir)
    global VIEWS_DIR
    VIEWS_DIR = sample_path
    
    if not os.path.exists(model_path):      
        os.mkdir(model_path)
    if not os.path.exists(sample_path):
        os.mkdir(sample_dir)

    rgb = np.load(f'{samples_path}{VAL_samples[index]}_image.npy')[2:250, 2:250, :3]
    depth = np.load(f'{samples_path}{VAL_samples[index]}_depth.npy')[2:250, 2:250]
    edge = np.load(f'{samples_path}{VAL_samples[index]}_gray_edges.npy')[2:250, 2:250]
    mask = np.load(f'{samples_path}0_mask.npy')

    fig1, ax1 = plt.subplots(1, 4)
    ax1[0].imshow(rgb)
    ax1[0].set_title('RGB')
    ax1[1].imshow((1-mask)*depth, cmap='viridis')
    ax1[1].set_title('masked depth')
    ax1[2].imshow(edge, cmap='gray')
    ax1[2].set_title('edges')
    ax1[3].imshow(depth, cmap='viridis')
    ax1[3].set_title('GT')
    for i in range(4):
        ax1[i].axis('off')
    
    gt_path = os.path.join(parent_dir, 'plots', 'GT', f'{split}_sample_{VAL_samples[index]}')
    global GT_DIR
    GT_DIR = gt_path

    if not os.path.exists(gt_path):      
        os.mkdir(gt_path)
    plt.savefig(f'{gt_path}/input_and_gt.pdf', format='pdf', bbox_inches='tight', dpi=1200)
    # plt.show()

    batch = {
        'rgb': torch.Tensor(rgb).permute(2, 0, 1).unsqueeze(0),
        'depth': torch.Tensor(depth).unsqueeze(0).unsqueeze(0),
        'mask': torch.Tensor(mask).unsqueeze(0).unsqueeze(0),
        'edges': torch.Tensor(edge).unsqueeze(0).unsqueeze(0)
    }
    with torch.no_grad():
        depth_pred = model(batch)

        # if evaluate == 'EdgeAttentionV2' or evaluate == 'EdgeAttention':
        #     (depth_pred, _) = depth_pred
    

    if results['hyper_params']['model name'] == 'EdgeAttentionModel' or evaluate == 'EdgeAttentionV2':
        depth_pred, _ = depth_pred
    if 'multiscale' in results['hyper_params'].keys() and results['hyper_params']['multiscale']:
        _, _, depth_pred = depth_pred

    completed_depth = batch['depth'] * (1 - batch['mask']) + depth_pred * batch['mask']
    completed_depth = completed_depth.cpu().detach().numpy()[0, 0]
    # plt.imshow(completed_depth, cmap='viridis')
    # plt.axis('off')
    # plt.show()

    gt = get_geometry(rgb, depth, mask)
    pred = get_geometry(rgb, completed_depth, mask)

    # custom_draw_geometry_with_key_callback(pred)
    # custom_draw_geometry_with_key_callback(pred)

    # save_viewpoints(pred)
    # custom_draw_geometry_with_custom_fov(pred, +180)
    o3d.visualization.draw_geometries([pred])


    # save_GT_pointcloud_rotation(gt)
    # concat_views(sample_path, evaluate)


    # custom_draw_geometry_with_rotation(pred)
    # concat_views(GT_DIR)
    # plot_3D(gt, pred)


    

