from data import goes16s3
import os
import numpy as np

from joblib import delayed, Parallel

directory = './data/training/'
cpu_count = 20
years = [2018, 2017]
#channels = [list(range(1,4)), list(range(1,9)), list(range(1,17)),]
channels = [list(range(1,9)), list(range(1,17)),]

for c in channels:
    example_directory = os.path.join(directory, '9Min-%iChannels' % len(c))
    if not os.path.exists(example_directory):
        os.makedirs(example_directory)

    goespytorch = goes16s3.NOAAGOESS3(save_directory='/nobackupp10/tvandal/data/goes16',
                                     channels=c, product='ABI-L1b-RadM')
    jobs = []
    for year in years:
        for day in np.arange(1,365):
            print('Year: {}, Day: {}'.format(year, day))
            jobs.append(delayed(goespytorch.write_pytorch_examples)(example_directory, year, day,
                                 force=False))

    # TODO: THIS IS A HACK, MAKE A WORKER FUNCTION
    try:
        Parallel(n_jobs=cpu_count)(jobs)
    except ValueError:
        continue
