import pandas as pd
import numpy as np
import pickle

def create_table(model_index, results_paths, split='val'):
    subset_scores = ['abs rel', 'rmse', 'delta 1.25', 'delta 1.25^2', 'delta 1.25^3']

    results_table = pd.DataFrame(columns=subset_scores, index=model_index)
    for (i, paths) in zip(model_index, results_paths):
        result_dicts = []
        convergence_steps = []
        for p in paths:
            file = open(p, 'rb')
            results = pickle.load(file)
            n_steps = results['model_path'].split('step=')[1][:7]
            n_steps = n_steps.split('.')[0]
            convergence_steps.append(int(n_steps))
            result_dicts.append(results)

        for s in subset_scores:
            values = tuple(results[split][0][s] for results in result_dicts)

            mean = np.mean(values).round(4)
            std = np.std(values).round(4)
            results_table.at[i, s] = f'{mean} ({std})'
            results_table.at[i, 'runs'] = len(values)
        results_table.at[i, 'avg #steps until convergence'] = np.mean(convergence_steps)
    print(f'{split} scores')
    print(results_table)
    # print(results_table.to_latex())

if __name__ == "__main__":
    split = 'val'
    baseline = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/Baseline_batch8_20.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/Baseline_batch8_21.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/Baseline_batch8_22.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/Baseline_batch8_23.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/Baseline_batch8_24.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/Baseline_batch8_25.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/Baseline_batch8_26.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/Baseline_batch8_27.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/Baseline_batch8_28.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/Baseline_batch8_29.pickle'
    ]

    gated_baseline = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedBaseline_batch8_20.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedBaseline_batch8_21.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedBaseline_batch8_22.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedBaseline_batch8_23.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedBaseline_batch8_24.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedBaseline_batch8_25.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedBaseline_batch8_26.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedBaseline_batch8_27.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedBaseline_batch8_28.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedBaseline_batch8_29.pickle',
    ]

    # deformable_baseline = [
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/DeformableBaseline_batch8_20.pickle',
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/DeformableBaseline_batch8_21.pickle',
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/DeformableBaseline_batch8_22.pickle',
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/DeformableBaseline_batch8_23.pickle'
    # ]

    deformable_baseline = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/DeformableBaseline10epochs_batch8_40.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/DeformableBaseline10epochs_batch8_41.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/DeformableBaseline10epochs_batch8_42.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/DeformableBaseline10epochs_batch8_44.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/DeformableBaseline10epochs_batch8_46.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/DeformableBaseline10epochs_batch8_48.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/DeformableBaseline10epochs_batch8_49.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/DeformableBaseline_batch8_20.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/DeformableBaseline_batch8_21.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/DeformableBaseline_batch8_22.pickle',
    ]

    # gateddeformable_baseline = [
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedDeformableBaseline_batch8_20.pickle',
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedDeformableBaseline_batch8_21.pickle',
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedDeformableBaseline_batch8_22.pickle',
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedDeformableBaseline_batch8_23.pickle',
    #     '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedDeformableBaseline_batch8_24.pickle'
    # ]

    gateddeformable_baseline = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedDeformableBaseline_batch8_40.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedDeformableBaseline_batch8_41.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedDeformableBaseline_batch8_42.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedDeformableBaseline_batch8_43.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedDeformableBaseline_batch8_44.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedDeformableBaseline_batch8_45.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedDeformableBaseline_batch8_46.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedDeformableBaseline_batch8_47.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedDeformableBaseline_batch8_48.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment1/GatedDeformableBaseline_batch8_49.pickle',
    ]

    model_index = ['regular', 'deformable', 'gated', 'gated+deformable']
    results_paths = [baseline, deformable_baseline, gated_baseline, gateddeformable_baseline]

    create_table(model_index, results_paths)