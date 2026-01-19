import synthetic_data as lc_data
import pandas as pd
from os.path import exists

SIZES=[10, 100] #, 1000, 10000]
SETUPS=[True, False]
REPS=16

def main():
    for disjoint in SETUPS:
        if disjoint:
            disjoint_text = 'disjoint'
        else:
            disjoint_text = 'nondisjoint'

        for size in SIZES:
            if size == 10000:
                bio=False
            else:
                bio=True
            full_values = []
            if not exists(f'./ablation_data/clustered_{disjoint_text}_data_{REPS}_reps_{size}_samps.csv'):
                for rep in range(REPS):
                    data = lc_data.run_simulation(rep, cluster_size=size, bio=bio, disjoint=disjoint)
                    full_values.extend(data)
                df = pd.concat(full_values)
                df.to_csv(f'/home/TomKerby/Research/local_corex_private/scripts/ablation_data/clustered_{disjoint_text}_data_{REPS}_reps_{size}_samps.csv')
            else:
                print(f'clustered_{disjoint_text}_data_{REPS}_reps_{size}_samps.csv already created.')

if __name__ == "__main__":
    main()