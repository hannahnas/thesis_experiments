import pandas as pd
import numpy as np
import pickle

if __name__ == "__main__":
    split = 'val'

    skipnet_single_enc = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SingleEncoderSkipNet_batch8_20.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SingleEncoderSkipNet_batch8_21.pickle',
        # '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SingleEncoderSkipNet_batch8_22.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SingleEncoderSkipNet_batch8_23.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SingleEncoderSkipNet_batch8_24.pickle',
        # '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SingleEncoderSkipNet_batch8_25.pickle',
        # '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SingleEncoderSkipNet_batch8_26.pickle',
        # '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SingleEncoderSkipNet_batch8_27.pickle',
        # '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SingleEncoderSkipNet_batch8_28.pickle',
        # '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SingleEncoderSkipNet_batch8_29.pickle'
    ]

    skipnet_dual_enc = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNet_batch8_20.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNet_batch8_21.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNet_batch8_22.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNet_batch8_23.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNet_batch8_24.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNet_batch8_25.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNet_batch8_26.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNet_batch8_27.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNet_batch8_28.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNet_batch8_29.pickle'
    ]
    
    skipnet_single_enc_edge = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SingleEncoderSkipNetWithEdge_batch8_20.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SingleEncoderSkipNetWithEdge_batch8_21.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SingleEncoderSkipNetWithEdge_batch8_22.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SingleEncoderSkipNetWithEdge_batch8_23.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SingleEncoderSkipNetWithEdge_batch8_24.pickle',
        # '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SingleEncoderSkipNetWithEdge_batch8_25.pickle',
    ]

    skipnet_dual_enc_depthedge = [
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

    skipnet_dual_enc_rgbedge = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithRGBEdges_batch8_20.pickle',
        # '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithRGBEdges_batch8_21.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithRGBEdges_batch8_22.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithRGBEdges_batch8_23.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithRGBEdges_batch8_24.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithRGBEdges_batch8_25.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithRGBEdges_batch8_26.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithRGBEdges_batch8_27.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithRGBEdges_batch8_28.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/SkipNetWithRGBEdges_batch8_29.pickle'
    ]

    edgeattention = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/EdgeAttentionModel_batch8_20.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/EdgeAttentionModel_batch8_21.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/EdgeAttentionModel_batch8_22.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/EdgeAttentionModel_batch8_23.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment2/EdgeAttentionModel_batch8_24.pickle'
    ]

    skipattention = [
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

    scores = ['abs rel', 'sq rel', 'rmse', 'rmse log', 'delta 1.25', 'delta 1.25^2', 'delta 1.25^3']
    subset_scores = ['abs rel', 'rmse', 'delta 1.25', 'delta 1.25^2', 'delta 1.25^3']
    index = ['Single Encoder Skipnet', 'Dual Encoder Skipnet', 'Single Encoder Skipnet with edge', 'Dual Encoder Skipnet depth with edge', 'Dual Encoder Skipnet rgb with edge', 'Edge Attention', 'Skip Attention']
    results_paths = [skipnet_single_enc, skipnet_dual_enc, skipnet_single_enc_edge, skipnet_dual_enc_depthedge, skipnet_dual_enc_rgbedge, edgeattention, skipattention]

    results_table = pd.DataFrame(columns=subset_scores, index=index)
    for (i, paths) in zip(index, results_paths):
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