import pandas as pd
import numpy as np
import pickle

if __name__ == "__main__":
    split = 'val'

    skipnet = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/SkipNet_batch8_lr0.0001_10.pickle'
    ]

    edgeattention = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/EdgeAttentionModel_batch8_lr0.0001_10.pickle'
    ]

    skipattention = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/GatedSkipAttention_batch8_lr0.0001_10.pickle'
    ]

    multiscale_skipattention = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/MultiScaleSkipAttention_batch8_lr0.0001_10.pickle'
    ]

    multiscale_skipattention_smooth = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/MultiScaleSkipAttentionWithSmoothness_batch8_lr0.0001_10.pickle'
    ]

    scores = ['abs rel', 'sq rel', 'rmse', 'rmse log', 'delta 1.25', 'delta 1.25^2', 'delta 1.25^3']
    subset_scores = ['abs rel', 'rmse', 'delta 1.25', 'delta 1.25^2', 'delta 1.25^3']
    index = ['SkipResNet', 'Edgeattention', 'Skipattention', 'Multi-scale Skipattention', 'Multi-scale Skipattention smooth']
    results_paths = [skipnet, edgeattention, skipattention, multiscale_skipattention, multiscale_skipattention_smooth]

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
            results_table1.at[i, s] = f'{mean}'# ({std})'
            # results_table1.at[i, 'runs'] = len(values)

    print(f'{split} scores')
    print(results_table1)
    print(results_table1.to_latex())