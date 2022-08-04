import pandas as pd
import numpy as np
import pickle

if __name__ == "__main__":
    split = 'val'

    skipnet = [
        '/home/hannah/Documents/Thesis/EdgeBasedInpainting/results/lisa_results/SkipNet_batch8_lr0.0001_0.pickle',
        '/home/hannah/Documents/Thesis/EdgeBasedInpainting/results/lisa_results/SkipNetWithEdges_batch8_lr0.0001_1.pickle'
    ]

    skipnet_edge_channel = [
        '/home/hannah/Documents/Thesis/EdgeBasedInpainting/results/lisa_results/SkipNetWithEdges_batch8_lr0.0001_0.pickle',
        '/home/hannah/Documents/Thesis/EdgeBasedInpainting/results/lisa_results/SkipNetWithEdges_batch8_lr0.0001_1.pickle'
    ]

    skipattention = [
        '/home/hannah/Documents/Thesis/EdgeBasedInpainting/results/lisa_results/GatedSkipAttention_batch8_lr0.0001_0.pickle',
        '/home/hannah/Documents/Thesis/EdgeBasedInpainting/results/lisa_results/GatedSkipAttention_batch8_lr0.0001_1.pickle',
        '/home/hannah/Documents/Thesis/EdgeBasedInpainting/results/lisa_results/GatedSkipAttention_batch8_lr0.0001_2.pickle'
    ]

    multiscale_skipattention = [
        '/home/hannah/Documents/Thesis/EdgeBasedInpainting/results/lisa_results/GatedMultiScaleSkipAttention(rescaledl1)_batch8_lr0.0001_1.pickle',
        '/home/hannah/Documents/Thesis/EdgeBasedInpainting/results/lisa_results/GatedMultiScaleSkipAttention(rescaledl1)_batch8_lr0.0001_2.pickle'
    ]

    multiscale_skipattention_smooth = [
        '/home/hannah/Documents/Thesis/EdgeBasedInpainting/results/lisa_results/GatedMultiScaleSkipAttentionWithSmoothness(lambda0.01)_batch8_lr0.0001_1.pickle',
        '/home/hannah/Documents/Thesis/EdgeBasedInpainting/results/lisa_results/GatedMultiScaleSkipAttentionWithSmoothness(lambda0.01)_batch8_lr0.0001_2.pickle'
    ]

    scores = ['abs rel', 'sq rel', 'rmse', 'rmse log', 'delta 1.25', 'delta 1.25^2', 'delta 1.25^3']
    subset_scores = ['abs rel', 'rmse', 'delta 1.25', 'delta 1.25^2', 'delta 1.25^3']
    index = ['SkipResNet', 'SkipResNet (edge)', 'Skipattention', 'Multi-scale Skipattention', 'Multi-scale Skipattention smooth']
    results_paths = [skipnet, skipnet_edge_channel, skipattention, multiscale_skipattention, multiscale_skipattention_smooth]

    results_table1 = pd.DataFrame(columns=subset_scores, index=index)
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
            results_table1.at[i, s] = f'{mean} ({std})'
            results_table1.at[i, 'runs'] = len(values)

    print(f'{split} scores')
    print(results_table1)
    # print(results_table1.to_latex())