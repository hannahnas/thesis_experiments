import pandas as pd
import numpy as np
import pickle


if __name__ == "__main__":
    split = 'val'
    baseline = [
    ]

    gated_baseline = [

    ]

    deformable_baseline = [

    ]

    gateddeformable_baseline = [
        
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