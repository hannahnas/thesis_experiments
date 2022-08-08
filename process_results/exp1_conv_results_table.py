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
    print(results_table.to_latex())

if __name__ == "__main__":
    split = 'val'
    baseline = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/Baseline_batch8_20.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/Baseline_batch8_21.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/Baseline_batch8_22.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/Baseline_batch8_23.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/Baseline_batch8_24.pickle'
    ]

    gated_baseline = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedBaseline_batch8_20.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedBaseline_batch8_21.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedBaseline_batch8_22.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedBaseline_batch8_23.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedBaseline_batch8_24.pickle'

    ]

    deformable_baseline = [

    ]

    gateddeformable_baseline = [
        
    ]

    model_index = ['regular', 'deformable', 'gated', 'gated+deformable']
    results_paths = [baseline, deformable_baseline, gated_baseline, gateddeformable_baseline]

    create_table(model_index, results_paths)