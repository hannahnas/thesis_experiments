import pickle
import numpy as np
import pandas as pd

def create_table(model_index, results_paths, split='val'):
    subset_scores = ['abs rel', 'rmse', 'delta 1.25', 'delta 1.25^2', 'delta 1.25^3']

    results_table = pd.DataFrame(columns=subset_scores, index=model_index)
    for (i, paths) in zip(model_index, results_paths):
        result_dicts = []
        convergence_steps = []
        for p in paths:
            file = open(p, 'rb')
            results = pickle.load(file)
            n_steps = results['model_path'].split('step=')[1][:5]
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
    print(results_table.to_latex())

if __name__ == "__main__":
    depth2depth = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment0/Depth2DeptInitialModel_batch8_20.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment0/Depth2DeptInitialModel_batch8_21.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment0/Depth2DeptInitialModel_batch8_22.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment0/Depth2DeptInitialModel_batch8_23.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment0/Depth2DeptInitialModel_batch8_24.pickle'
    ]

    channelwise = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment0/ChannelWiseInitialModel_batch8_20.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment0/ChannelWiseInitialModel_batch8_21.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment0/ChannelWiseInitialModel_batch8_22.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment0/ChannelWiseInitialModel_batch8_23.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment0/ChannelWiseInitialModel_batch8_24.pickle'
    ]

    dual_encoder = [
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment0/DualEncoderInitialModel_batch8_20.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment0/DualEncoderInitialModel_batch8_21.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment0/DualEncoderInitialModel_batch8_22.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment0/DualEncoderInitialModel_batch8_23.pickle',
        '/home/hannah/Documents/Thesis/thesis_experiments/results/experiment0/DualEncoderInitialModel_batch8_24.pickle'
    ]

    model_index = ['depth only', 'single encoder', 'dual encoder']
    results_paths = [depth2depth, channelwise, dual_encoder]

    create_table(model_index, results_paths)
