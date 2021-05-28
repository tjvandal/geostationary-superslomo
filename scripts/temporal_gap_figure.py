import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import matplotlib
matplotlib.rc('text', usetex=True)

import seaborn as sns
sns.set(style="whitegrid")
sns.set_context("paper", font_scale=2.5)

matplotlib.rc("text", usetex=True)
import matplotlib.pyplot as plt

import comparison

def main():
    '''
    Make plot such that x=Temporal gap, y=RMSE, lines=methods
    Band is static, selected to be 13
    '''
    T_range = [5,10,15,20,25,30,35,40,45,]
    N_samples = 20
    band = 13

    stat_file = '.tmp/temporal-stats.pkl'
    force = False

    if (os.path.exists(stat_file)) and (not force):
        all_stats = pd.read_pickle(stat_file)
        all_stats = all_stats.reset_index()
    else:
        pairs = comparison.random_dayhours(N_samples, 2019)
        jobs = []
        for p in pairs:
            for T in T_range:
                jobs.append(delayed(comparison.example_statistics)(2019, p[0], p[1], band, T=T))

        print("number of jobs", len(jobs))
        collect_stats = Parallel(n_jobs=4)(jobs)
        #collect_stats.append(p_stats)
        all_stats = pd.concat(collect_stats)
        all_stats = all_stats.reset_index()
        all_stats.to_pickle(stat_file)
    
    all_stats['Model'][all_stats['Model'] == 'Slomo'] = 'SSM-T'
    all_stats['Model'][all_stats['Model'] == 'Slomo-MS'] = 'SSM-TMS'
    all_stats['Model'][all_stats['Model'] == 'Slomo-G'] = 'SSM-G'

    g = sns.catplot(x='T', y='PSNR', kind='point', hue='Model', data=all_stats, aspect=1.5,
                height=6, capsize=.05, legend=False)
    plt.xlabel("Temporal Gap Between Images")
    plt.legend(loc='upper right')
    g.despine(left=True)
    plt.savefig("figures/temporal_gap.pdf", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
