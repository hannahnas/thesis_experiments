import pandas as pd
import numpy as np
import pickle

def create_table(model_index, results_paths, split='val'):
    subset_scores = ['abs rel', 'rmse', 'delta 1.25', 'delta 1.25^2', 'delta 1.25^3']

    results_table = pd.DataFrame(columns=subset_scores, index=model_index)
    for (i, paths) in zip(model_index, results_paths):
        result_dicts = []
        for p in paths:
            file = open(p, 'rb')
            results = pickle.load(file)
            result_dicts.append(results)

        for s in subset_scores:
            values = tuple(results[split][0][s] for results in result_dicts)

            mean = np.mean(values).round(4)
            std = np.std(values).round(4)
            results_table.at[i, s] = f'{mean} ({std})'
            results_table.at[i, 'runs'] = len(values)
    print(f'{split} scores')
    print(results_table)
    # print(results_table.to_latex())




if __name__ == "__main__":
    split = 'val'

    skipattn = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/GatedSkipAttention_batch8_30.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/GatedSkipAttention_batch8_31.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/GatedSkipAttention_batch8_32.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/GatedSkipAttention_batch8_33.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/GatedSkipAttention_batch8_34.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/GatedSkipAttention_batch8_35.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/GatedSkipAttention_batch8_36.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/GatedSkipAttention_batch8_37.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/GatedSkipAttention_batch8_38.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/GatedSkipAttention_batch8_39.pickle'
    ]
    
    skipattn_smooth = [
        # '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/SkipAttentionSmooth_batch8_20.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/SkipAttentionSmooth_batch8_21.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/SkipAttentionSmooth_batch8_22.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/SkipAttentionSmooth_batch8_23.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/SkipAttentionSmooth_batch8_24.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/SkipAttentionSmooth_batch8_25.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/SkipAttentionSmooth_batch8_26.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/SkipAttentionSmooth_batch8_27.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/SkipAttentionSmooth_batch8_28.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/SkipAttentionSmooth_batch8_29.pickle',
    ]

    # skipattn_multi_scale = [
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScaleSkipAttention_batch8_20.pickle',
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScaleSkipAttention_batch8_21.pickle',
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScaleSkipAttention_batch8_22.pickle',
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScaleSkipAttention_batch8_23.pickle',
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScaleSkipAttention_batch8_24.pickle'
    # ]

    skipattn_multiscale1x1 = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScale1x1SkipAttention_batch8_50.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScale1x1SkipAttention_batch8_51.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScale1x1SkipAttention_batch8_52.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScale1x1SkipAttention_batch8_53.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScale1x1SkipAttention_batch8_54.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScale1x1SkipAttention_batch8_55.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScale1x1SkipAttention_batch8_56.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScale1x1SkipAttention_batch8_57.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScale1x1SkipAttention_batch8_58.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScale1x1SkipAttention_batch8_59.pickle',
    ]

    # skipattn_multi_scale_smooth = [
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScaleSkipAttentionWithSmoothness_batch8_20.pickle',
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScaleSkipAttentionWithSmoothness_batch8_21.pickle',
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScaleSkipAttentionWithSmoothness_batch8_22.pickle',
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScaleSkipAttentionWithSmoothness_batch8_23.pickle',
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScaleSkipAttentionWithSmoothness_batch8_24.pickle'
    # ]

    skipattn_multiscale1x1_smooth = [ 
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScale1x1SkipAttentionWithSmoothness_batch8_50.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScale1x1SkipAttentionWithSmoothness_batch8_51.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScale1x1SkipAttentionWithSmoothness_batch8_52.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScale1x1SkipAttentionWithSmoothness_batch8_53.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScale1x1SkipAttentionWithSmoothness_batch8_54.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScale1x1SkipAttentionWithSmoothness_batch8_55.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScale1x1SkipAttentionWithSmoothness_batch8_56.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScale1x1SkipAttentionWithSmoothness_batch8_57.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScale1x1SkipAttentionWithSmoothness_batch8_58.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScale1x1SkipAttentionWithSmoothness_batch8_59.pickle',
    ]

    skipnet_edge = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithEdges_batch8_20.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithEdges_batch8_21.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithEdges_batch8_22.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithEdges_batch8_23.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithEdges_batch8_24.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithEdges_batch8_25.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithEdges_batch8_26.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithEdges_batch8_27.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithEdges_batch8_28.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithEdges_batch8_29.pickle'
    ]
    # skipnet_edge_ms = [
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscale_batch8_30.pickle',
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscale_batch8_31.pickle',
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscale_batch8_32.pickle',
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscale_batch8_33.pickle',
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscale_batch8_33.pickle'
    # ]
    skipnet_edge_ms1x1 = [ 
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscale1x1_batch8_50.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscale1x1_batch8_51.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscale1x1_batch8_52.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscale1x1_batch8_53.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscale1x1_batch8_54.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscale1x1_batch8_55.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscale1x1_batch8_56.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscale1x1_batch8_57.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscale1x1_batch8_58.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscale1x1_batch8_59.pickle',
    ]

    skipnet_edge_smooth = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithSmoothness_batch8_20.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithSmoothness_batch8_21.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithSmoothness_batch8_22.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithSmoothness_batch8_23.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithSmoothness_batch8_24.pickle',
        # '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithSmoothness_batch8_30.pickle',
        # '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithSmoothness_batch8_31.pickle',
        # '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithSmoothness_batch8_32.pickle',
        # '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithSmoothness_batch8_33.pickle',
        # '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithSmoothness_batch8_34.pickle',
    ]

    # skipnet_edge_ms_smooth = [
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscaleSmoothness_batch8_20.pickle',
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscaleSmoothness_batch8_21.pickle',
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscaleSmoothness_batch8_22.pickle',
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscaleSmoothness_batch8_23.pickle',
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscaleSmoothness_batch8_24.pickle'
    # ]

    skipnet_edge_ms1x1_smooth = [ 
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWith1x1MultiscaleSmoothness_batch8_50.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWith1x1MultiscaleSmoothness_batch8_51.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWith1x1MultiscaleSmoothness_batch8_52.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWith1x1MultiscaleSmoothness_batch8_53.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWith1x1MultiscaleSmoothness_batch8_54.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWith1x1MultiscaleSmoothness_batch8_55.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWith1x1MultiscaleSmoothness_batch8_56.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWith1x1MultiscaleSmoothness_batch8_57.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWith1x1MultiscaleSmoothness_batch8_58.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWith1x1MultiscaleSmoothness_batch8_59.pickle'
    ]

    subset_scores = ['abs rel', 'rmse', 'delta 1.25', 'delta 1.25^2', 'delta 1.25^3']
    model_index = [
        'Skip Attention',
        # 'Skip Attention + ms', 
        'Skip Attention + ms 1x1',
        'Skip attention + smooth', 
        # 'Skip attention + ms + smooth', 
        'Skip Attention + ms 1x1 + smooth',
        'Skipnet (edge)', 
        # 'Skipnet (edge) + ms', 
        'Skipnet (edge) + ms 1x1',
        'Skipnet (edge) + smooth', 
        # 'Skipnet (edge) + ms + smooth',
        'Skipnet (edge) + ms 1x1 + smooth'
        ]

    results_paths = [
        skipattn,
        # skipattn_multi_scale, 
        skipattn_multiscale1x1,
        skipattn_smooth, 
        # skipattn_multi_scale_smooth,
        skipattn_multiscale1x1_smooth,
        skipnet_edge,
        # skipnet_edge_ms,
        skipnet_edge_ms1x1,
        skipnet_edge_smooth,
        # skipnet_edge_ms_smooth,
        skipnet_edge_ms1x1_smooth
        ]

    create_table(model_index, results_paths, split='val')
    print('')
