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
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/GatedSkipAttention_batch8_34.pickle'
    ]
    
    skipattn_smooth = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/SkipAttentionSmooth_batch8_21.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/SkipAttentionSmooth_batch8_22.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/SkipAttentionSmooth_batch8_23.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/SkipAttentionSmooth_batch8_24.pickle'
    ]

    skipattn_multi_scale = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScaleSkipAttention_batch8_20.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScaleSkipAttention_batch8_21.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScaleSkipAttention_batch8_22.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScaleSkipAttention_batch8_23.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScaleSkipAttention_batch8_24.pickle'
    ]

    skipattn_multi_scale_smooth = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScaleSkipAttentionWithSmoothness_batch8_20.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScaleSkipAttentionWithSmoothness_batch8_21.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScaleSkipAttentionWithSmoothness_batch8_22.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScaleSkipAttentionWithSmoothness_batch8_23.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/MultiScaleSkipAttentionWithSmoothness_batch8_24.pickle'
    ]

    skipnet_edge = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithEdges_batch8_20.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithEdges_batch8_21.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithEdges_batch8_22.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithEdges_batch8_23.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithEdges_batch8_24.pickle'
    ]
    skipnet_edge_ms = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscale_batch8_30.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscale_batch8_31.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscale_batch8_32.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscale_batch8_33.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscale_batch8_33.pickle'
    ]

    skipnet_edge_smooth = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithSmoothness_batch8_20.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithSmoothness_batch8_21.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithSmoothness_batch8_22.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithSmoothness_batch8_23.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithSmoothness_batch8_24.pickle'
    ]

    skipnet_edge_ms_smooth = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscaleSmoothness_batch8_20.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscaleSmoothness_batch8_21.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscaleSmoothness_batch8_22.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscaleSmoothness_batch8_23.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment3/DualEncoderSkipnetEdgeWithMultiscaleSmoothness_batch8_24.pickle'
    ]

    subset_scores = ['abs rel', 'rmse', 'delta 1.25', 'delta 1.25^2', 'delta 1.25^3']
    model_index = [
        'Skip Attention',
        'Skip Attention + ms', 
        'Skip attention + smooth', 
        'Skip attention + ms + smooth', 
        'Skipnet (edge)', 
        'Skipnet (edge) + ms', 
        'Skipnet (edge) + smooth', 
        'Skipnet (Edge) + ms + smooth'
        ]
    results_paths = [
        skipattn,
        skipattn_multi_scale, 
        skipattn_smooth, 
        skipattn_multi_scale_smooth,
        skipnet_edge,
        skipnet_edge_ms,
        skipnet_edge_smooth,
        skipnet_edge_ms_smooth
        ]

    create_table(model_index, results_paths, split='val')
    print('')
