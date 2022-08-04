import pandas as pd
import numpy as np
import pickle


if __name__ == "__main__":
    split = 'val'
    baseline = [
        '/home/hannah/Documents/Thesis/EdgeBasedInpainting/results/lisa_results/Baseline_batch8_lr0.0001_3.pickle',
        '/home/hannah/Documents/Thesis/EdgeBasedInpainting/results/lisa_results/Baseline_batch8_lr0.0001_4.pickle',
        '/home/hannah/Documents/Thesis/EdgeBasedInpainting/results/lisa_results/Baseline_batch8_lr0.0001_5.pickle'
    ]

    gated_baseline = [
        '/home/hannah/Documents/Thesis/EdgeBasedInpainting/results/lisa_results/GatedBaseline_batch8_lr0.0001_3.pickle',
        '/home/hannah/Documents/Thesis/EdgeBasedInpainting/results/lisa_results/GatedBaseline_batch8_lr0.0001_4.pickle',
        '/home/hannah/Documents/Thesis/EdgeBasedInpainting/results/lisa_results/GatedBaseline_batch8_lr0.0001_5.pickle',
        # '/home/hannah/Documents/Thesis/EdgeBasedInpainting/results/lisa_results/GatedBaseline_batch4_lr0.0001_6.pickle'
    ]

    deformable_baseline = [
        '/home/hannah/Documents/Thesis/EdgeBasedInpainting/results/lisa_results/DeformableBaseline_batch8_lr0.0001_3.pickle',
        '/home/hannah/Documents/Thesis/EdgeBasedInpainting/results/lisa_results/DeformableBaseline_batch8_lr0.0001_4.pickle',
        '/home/hannah/Documents/Thesis/EdgeBasedInpainting/results/lisa_results/DeformableBaseline_batch8_lr0.0001_7.pickle'
    ]

    gateddeformable_baseline = [
        '/home/hannah/Documents/Thesis/EdgeBasedInpainting/results/lisa_results/GatedDeformableBaseline_batch8_lr0.0001_5.pickle',
        '/home/hannah/Documents/Thesis/EdgeBasedInpainting/results/lisa_results/GatedDeformableBaseline_batch8_lr0.0001_3.pickle',
        '/home/hannah/Documents/Thesis/EdgeBasedInpainting/results/lisa_results/GatedDeformableBaseline_batch8_lr0.0001_6.pickle'
    ]

    scores = ['abs rel', 'sq rel', 'rmse', 'rmse log', 'delta 1.25', 'delta 1.25^2', 'delta 1.25^3']
    subset_scores = ['abs rel', 'rmse', 'delta 1.25', 'delta 1.25^2', 'delta 1.25^3']
    index = ['regular', 'deformable', 'gated', 'gated+deformable']
    results_paths = [baseline, deformable_baseline, gated_baseline, gateddeformable_baseline]

    results_table1 = pd.DataFrame(columns=subset_scores, index=index)
    for (i, paths) in zip(index, results_paths):
        result_dicts = []
        for p in paths:
            file = open(p, 'rb')
            results = pickle.load(file)
            result_dicts.append(results)

        for s in subset_scores:
            values = tuple(results[split][0][s] for results in result_dicts)
            # print(values)
            mean = np.mean(values).round(4)
            std = np.std(values).round(4)
            results_table1.at[i, s] = f'{mean} ({std})'
            results_table1.at[i, 'runs'] = len(values)
    print(f'{split} scores')
    print(results_table1)
    print(results_table1.to_latex())