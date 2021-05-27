# Geostationary Temporal Interpolation with Optical Flow

Depends on NOAA AWS S3 ABI data.  Trains and tests on mesoscale, inference on CONUS and full disk.

## Dependencies

`conda create env -f environment.yaml`

Ensure your AWS S3 aws_access_key and aws_secret_access_key are set. 

## Download data and/or read from NAS

The file `data/goes16s3.py` contains two classes and functions to download and process training and test datsets.

`NOAAGOESS3`: Connects to NOAAs AWS S3 bucket to download data locally, read the files accordingly, and generate pytorch examples. This should be split into two classes, one that downloads the data and other that reads the local files. <br>
`GOESDataset`: Pytorch dataset object for training. 

`download_data(test, n_jobs)`: Downloads training and test data.  Every 5 days of 2017 and 2018 for training and 2019 for testing. 

By default, this script will download train and test datasets as needed. Currently the local data directory needs to be updated for your system (this needs to be fixed). <br>
Run `python data/goes16s3.py`

## Make training patches (years 2017, 2018)

Generating training patches from the raw data is done in parallel on a single node with the script `python make_training_data.py` or by submitting a job, `qsub make_training_data.pbs`.  

## Perform Bayesian Optimization to find Hyper-parameters

Bayesian Optimization is used to select hyperparameters using the Ax library and Pytorch. `bayesopt_training.py` will find hyper-parameters for a specified set of channels, `n_channels`, and single- or multi-variate, `multivariate`. Multiple experiments can be ran in parallel on NAS's v100 queue on sky_gpu using the pbs script `./mpi.bayesopt_training.pbs`. 

## Training models and experiments

`train_net.py` contains the base training methods for interpolation model, include the flow and interpolation convolutional neural networks. Currently, the hyperparameters are hardcoded, this may need to change. 

To train single experiment: `python ./train_net.py --epochs 50 --n_channels 8 --multivariate`<br>
To train set of experiments: Define parameterizations in `./mpi.train_net.py` and submit NAS job `qsub ./mpi.train_net.pbs`

## Run inference on test set (year 2019)

Perform inference and store outputs for analysis. A linear interpolation prediction is also made, all predictions are loaded into a single .nc file. 

Single model: `python ./inference_test.py`<br>
Parallelized by day: `./mpi.inference_test.py` submitted on NAS as `qsub ./mpi.inference_test.py`

## Analysis of Test Set

After performing inference on the test set for you're selected n_channels (we used 3 and 8 for the interpolation paper), we generate tables and figures of the results using code in `./notebooks/`,  `python ./notebooks/testing_tables.py`.

TODOs
1. Clean of analysis scripts
1. Split NOAAGOESS3 into multiple classes (this requires downstream changes). 
