import numpy as np
import pandas as pd
from evaluation import forecast_test_future

from darts import TimeSeries
from myTimeSeries import TimeSeriesList

def get_all_blocks(train, n_in, n_out):
    """
    train (TimeSeries): list of all possible blocks of the series with length n_in + n_out

    PARAMETERS:
    n_in (int): input length for the model
    n_out (int): output length for the model
    
    RETURNS: 
    a list with all possible blocks of train with length n_in+n_out
    """
    l = n_in + n_out
    blocks = []
    for t in train:
        n = len(t) - l
        for i in range(n):
            blocks.append(t[i:i+l])

    return blocks

def resample_with_replacement(blocks, n=None):
    """
    choose n random blocks out of 'blocks' (with replacement, i.e. same block can be chosen multiple times)
    
    PARAMETERS:
    blocks (list of TimeSeries): list of all possible blocks
    n (int): number of blocks to choose (if None choose same amount as in 'blocks')
    
    RETURNS:
    list of resampled blocks
    """
    if n is None:
        n = len(blocks)
    bootstrap_sample = []
    while len(bootstrap_sample) < n:
        r = np.random.randint(0, len(blocks))
        bootstrap_sample.append(blocks[r])
    return bootstrap_sample
    
    
def mean_and_variance(bootstrap_predictions):
    """
    calculate mean and variance from predictions
    """
    n_bootstrap = len(bootstrap_predictions)

    mean = bootstrap_predictions[0] + bootstrap_predictions[1]
    for i in range(2, n_bootstrap):
        mean += bootstrap_predictions[i]
    mean = mean / n_bootstrap

    variance = (bootstrap_predictions[0] - mean)**2
    for i in range(1, n_bootstrap):
        variance += (bootstrap_predictions[i] - mean)**2
    variance = variance / (n_bootstrap-1)

    return mean, variance


def get_upper_lower(alpha, predictions):
    lower_bounds = []
    upper_bounds = []
    mean = []
    for i in range(predictions[0].len()): # iterate over different time series
        l_b = []
        u_b = []
        time = predictions[0][i].time_index
        for a in alpha:
            lower = np.percentile(TimeSeriesList(predictions).values(), 100*(a/2), axis=0)[i]
            upper = np.percentile(TimeSeriesList(predictions).values(), 100*(1-a/2), axis=0)[i]
            l_b.append(TimeSeriesList.from_times_and_values(time, lower))
            u_b.append(TimeSeriesList.from_times_and_values(time, upper))
        lower_bounds.append(TimeSeriesList(l_b))
        upper_bounds.append(TimeSeriesList(u_b))
        m = np.percentile(TimeSeriesList(predictions).values(), 50, axis=0)[i]
        mean.append(TimeSeriesList.from_times_and_values(time, m))
    return TimeSeriesList(mean), TimeSeriesList(lower_bounds), TimeSeriesList(upper_bounds)

def block_bootstrap_(x, run_model, hyperparameters, n_bootstrap, n_pred, alphas=[0.05], x_transfer = None):
    """
    block bootstrap for calculating the prediction interval. Resample blocks the training set with replacement n_bootstrap times. Each time new model with new prediction. Range of predictions gives prediction interval 
    
    PARAMETERS:
    x (myTimeSeries, or TimeSeriesList): series on which model is trained
    run_model (function): model from models.py
    hyperparameters (dict): dictionary with hyperparameters
    n_bootstrap (int): number of models trained with bootstrap, should be large (i.e. > 100)
    n_pred (int): number of timesteps to predict
    alphas (list of floats): list of the percentiles for the interval ([0.05, 0.1] gives 95% interval and 90% interval)
    x_transfer (optional, myTimeSeries or TimeSeriesList): if the same model is used also to predict a different time series (model is fitted only on x but also predicts on x_transfer) 
    
    RETURNS:
    mean: mean of all predictions from bootstrapping
    lower: alpha/2-percentile of all predictions (lower end of interval)
    upper: (1-alpha/2)-percentile of all predictions (upper end of interval)
    mean_transfer, lower_transfer, upper_transfer: same but for x_transfer (or None)
    """
    n_in = hyperparameters['n_in']
    n_out = hyperparameters['n_out']

    train_block = get_all_blocks(x.train, n_in, n_out)

    bootstrap_predictions = []
    bootstrap_predictions_transfer = []
    for i in range(n_bootstrap):
        print('\n bootstrap model ', i)
        # get a new training set (resample with replacement)
        bootstrapped_train = resample_with_replacement(train_block)
        # train on the x-series the model and forecast on it
        model_bootstrap = run_model(train=bootstrapped_train, hyperparameters=hyperparameters)
        bootstrap_predictions.append(forecast_test_future(x, model_bootstrap, n_pred, scale_back=True))

        if(x_transfer is not None):
            # used the before trained model to forecast on a different series (optional)
            bootstrap_predictions_transfer.append(forecast_test_future(x_transfer, model_bootstrap, n_pred, scale_back=True))

    mean, lower, upper = get_upper_lower(alphas, bootstrap_predictions)
    if(x_transfer is not None):
        mean_transfer, lower_transfer, upper_transfer = get_upper_lower(alphas, bootstrap_predictions_transfer)
    else:
        mean_transfer = None
        lower_transfer = None
        upper_transfer = None

    return mean, lower, upper, mean_transfer, lower_transfer, upper_transfer


def save_results(direct, mean, lower, upper, savingpath, names, alpha, addition=''):
    """
    save results from bootstrapping to .txt file
    
    direct, mean, lower, upper all time series we want to save in txt file with values and times
    savingpath (str): where to save the .txt files
    names (list of str): different series (e.g. sunspot and tsi) -> are saved in different files
    alpha (list of floats): percentile of the intervals (lower and upper) 
    addition (str): any additional information for file name
    """
    # save results to .txt file
    for i, n in enumerate(names):
        dataframe = pd.DataFrame({'time test': mean[i][0].time_index, 'direct test': direct[i][0].values().reshape(-1), 'mean test': mean[i][0].values().reshape(-1)})
        for j,a in enumerate(alpha):
            dataframe['lower test ' + str(a)] = lower[i][j][0].values().reshape(-1)
            dataframe['upper test ' + str(a)] = upper[i][j][0].values().reshape(-1)
        dataframe['time future'] = mean[i][1].time_index
        dataframe['direct future'] = direct[i][1].values().reshape(-1)
        dataframe['mean future'] = mean[i][1].values().reshape(-1)
        for j, a in enumerate(alpha):
            dataframe['lower future ' + str(a)] = lower[i][j][1].values().reshape(-1)
            dataframe['upper future ' + str(a)] = upper[i][j][1].values().reshape(-1)
        
        dataframe.to_csv(savingpath + addition + names[i] + '_bootstrapping_results.txt', sep=';')
    

