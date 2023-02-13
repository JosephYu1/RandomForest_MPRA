# author:       Joseph Yu
# file:         randomforest.py
# description:  This script performs training and testing of random forest models

import numpy as np
import os
import sys, traceback
import argparse
import datetime
import math
from functools import partial
from multiprocessing import Pool

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import dataload


###
#   helper functions
###
def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    
    return x

    
###
#   arguments
###
arg_parser = argparse.ArgumentParser(descriptioin="Train and test Random Forest Model for MPRA Activity")

arg_parser.add_argument("--no_sample", type='store_true', default=False, 
                        help='use same test training set as DragoNN-MPRA; default=False')

arg_parser.add_argument("--test_size", type=restricted_float, default=0.2, 
                        help='percentage of data sampled for testing; default=0.2')

arg_parser.add_argument("--sample_random_state", type=int, default=22,
                        help='train_test_split: random_state argument; default=22')

arg_parser.add_argument("--n_jobs", type=int, default=None, 
                        help='RandomForetRegressor: The number of jobs to run in parallel; default=None (1)')

arg_parser.add_argument("--model_random_state", type=int, default=None, 
                        help='RandomForestRegressor: random_state argument for random forest regressor; default=None')

arg_parser.add_argument("--n_estimators", type=int, default=100, 
                        help = 'RandomForestRegressor: n_estimators argument for random forest regressor; default=100')

arg_parser.add_argument("--max_depth", type=int, default=None, 
                        help='RandomForestRegressor: max_depth argument for random forest regressor; default=None')


args = arg_parser.parse_args()


# saving parameters
NO_SAMPLE = args.no_sample
TEST_SIZE = args.test_size
SAMPLE_RANDOM_STATE = args.sample_random_state
N_JOBS = args.n_jobs
MODEL_RANDOM_STATE = args.model_random_state
N_ESTIMATORS = args.n_estimators
MAX_DEPTH = args.max_depth


###
#   Sampling
###
def sampledata(dataX, dataY, sample_test_size, sample_random_state):

    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, 
                                                        test_size=sample_test_size, 
                                                        random_state=sample_random_state)

    result = {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}

    return result


###
#   Flatten one-hot
###
def flatten(data):
    nsamples, nx, ny = data.shape
    return data.reshape((nsamples, nx*ny))


###
#   RandomForestRegressor
###
def train():


def test():



###
#   main
###
def main(argv):
    data_dict = dataloader.load_file()

    model_data = None
    if NO_SAMPLE is False: 
        all_X = np.concatenate((data_dict['training_X'], data_dict['test_X'], data_dict['validation_X']), axis=0)
        all_Y = np.concatenate((data_dict['training_Y'], data_dict['test_Y'], data_dict['validation_Y']), axis=0)
     
        model_data = sampledata(all_X, all_Y, TEST_SIZE, SAMPLE_RANDOM_STATE)

    else:
        model_data = {"x_train": data_dict['training_X'], "x_test": data_dict['test_X'], 
                      "y_train": data_dict['training_Y'], "y_test": data_dict['test_Y']}

    model_data['x_train'] = flatten(model_data['x_train'])
    model_data['x_test'] = flatten(model_data['x_test'])

    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, 
                                  max_depth=MAX_DEPTH, 
                                  n_jobs=N_JOBS, 
                                  random_state=MODEL_RANDOM_STATE)

    model.fit(model_data['x_train'], model_data['y_train'])
    score = model.score(model_data['x_train'], model_data['y_train'])

    prediction = model.predict(model_data['x_test'])

    mse = mean_squared_error(model_data['y_test'], prediction)

    # write to output file
    # output_file_name = 
    # with open("")

    print("R-squared\tMSE\tRMSE")
    print(score, "\t", mse, "\t", mse*(1/2.0))


if __name__ == "__main__":
    main(sys.argv[1:])
