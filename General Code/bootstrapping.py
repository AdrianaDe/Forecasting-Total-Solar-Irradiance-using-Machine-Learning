import numpy as np
import pandas as pd
from evaluation import forecast_test_future

from darts import TimeSeries
from myTimeSeries import TimeSeriesList

def get_all_blocks(train, n_in, n_out):
    """
    set self.blocks and self.train_blocks
    train (TimeSeries): list of all possible blocks of the series with length n_in + n_out

    PARAMETERS:
    n_in (int): input length for the model
    n_out (int): output length for the model
    """
    l = n_in + n_out
    blocks = []
    for t in train:
        n = len(t) - l
        for i in range(n):
            blocks.append(t[i:i+l])

    return blocks

def add_residual(train_block, residual):
    n_blocks = len(residual) # number of blocks in one bootstrap sample
    n_out = len(residual[0])

    block = []
    for j in range(n_blocks):
        x = train_block[j][:-n_out]
        y_val = train_block[j][-n_out:].values().reshape(-1) + residual[j]
        y = TimeSeries.from_times_and_values(train_block[j][-n_out:].time_index, y_val)
        block.append(x.append(y))
    return block

def resample_with_replacement(blocks, n=None):
    if n is None:
        n = len(blocks)
    bootstrap_sample = []
    while len(bootstrap_sample) < n:
        r = np.random.randint(0, len(blocks))
        bootstrap_sample.append(blocks[r])
    return bootstrap_sample
    
    
def mean_and_variance(bootstrap_predictions):
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
    
    
def block_bootstrap(x, run_model, hyperparameters, n_bootstrap, n_pred, x_transfer = None, savingpath=None, names=None, addition='', names_transf=None):
    n_in = hyperparameters['n_in']
    n_out = hyperparameters['n_out']

    train_block = get_all_blocks(x.train, n_in, n_out)

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

    mean, variance = mean_and_variance(bootstrap_predictions)
    if(x_transfer is not None):
        mean_transfer, variance_transfer = mean_and_variance(bootstrap_predictions_transfer)
    else:
        mean_transfer = None
        variance_transfer = None

    # save results to .txt file
    if savingpath is not None:
        for i, n in enumerate(names):
            dataframe = pd.DataFrame({'time test': mean[i][0].time_index, 'mean test': mean[i][0].values().reshape(-1), 'variance test': variance[i][0].values().reshape(-1),
                            'time future': mean[i][1].time_index, 'mean test': mean[i][1].values().reshape(-1), 'variance future': variance[i][1].values().reshape(-1)})
            dataframe.to_csv(savingpath + addition + names[i] + '_bootstrapping_results(n='+str(n_bootstrap)+').txt', sep=';')
        
        for i, n in enumerate(names_transf):
            dataframe=pd.DataFrame({'time test': mean_transfer[i][0].time_index, 'mean test': mean_transfer[i][0].values().reshape(-1), 'variance test': variance_transfer[i][0].values().reshape(-1),
                                    'time future': mean_transfer[i][1].time_index, 'mean test': mean_transfer[i][1].values().reshape(-1), 'variance future': variance_transfer[i][1].values().reshape(-1)})
            dataframe.to_csv(savingpath + addition + names_transf[i] + '_transfer_bootstrapping_results(n='+str(n_bootstrap)+').txt', sep=';')
            
    return mean, variance, mean_transfer, variance_transfer
    
    
    
def resuidual_bootstrapping(run_model, series, hyperparameters, n_bootstrap, n_pred, alpha = [0.05], series_transfer=None, model_before_trained=None, savingpath=None):
    if model_before_trained is None:
        model = run_model(train = series.train, hyperparameters=hyperparameters)
    else:
        model = model_before_trained
    train_block = get_all_blocks(series.train, hyperparameters['n_in'], hyperparameters['n_out'])

    # calculate predictions on training set
    train_block_x = []
    for t in train_block:
        train_block_x.append(t[:hyperparameters['n_in']])
    prediction_block_y = model.predict(series=train_block_x, n=hyperparameters['n_out'])

    # calculate residual on training set
    residual = []
    for p in prediction_block_y:
        residual.append((p - series.train[0][p.start_time():p.end_time()]).values().reshape(-1))

    # bootstrap residuals
    bootstrap_predictions = []
    bootstrap_predictions_transfer = []
    for i in range(n_bootstrap):
        print('\n bootstrap model ', i)
        residual_bootstrapped = resample_with_replacement(residual)
        resampled_train = add_residual(train_block, residual_bootstrapped)
        model_bootstrap = run_model(train=resampled_train, hyperparameters=hyperparameters)
        bootstrap_predictions.append(forecast_test_future(x=series, model=model_bootstrap, n_pred=n_pred, scale_back=True))
        if series_transfer is not None:
            bootstrap_predictions_transfer.append(forecast_test_future(x=series_transfer, model=model_bootstrap, n_pred=n_pred, scale_back=True))

    # calculate the intervals (from percentile)
    lower_bounds = []
    upper_bounds = []
    mean = []
    for i in range(bootstrap_predictions[0].len()): # iterate over different time series
        l_b=[]
        u_b = []
        time = bootstrap_predictions[0][i].time_index
        for a in alpha:
            lower = np.percentile(TimeSeriesList(bootstrap_predictions).values(), 100*(a/2), axis=0)[i] # .reshape(2,-1)
            upper = np.percentile(TimeSeriesList(bootstrap_predictions).values(), 100*(1-a/2), axis=0)[i]# .reshape(2,-1)
            l_b.append(TimeSeriesList.from_times_and_values(time, lower))
            u_b.append(TimeSeriesList.from_times_and_values(time, upper))

        lower_bounds.append(TimeSeriesList(l_b))
        upper_bounds.append(TimeSeriesList(u_b))
        m = np.percentile(TimeSeriesList(bootstrap_predictions).values(), 50, axis=0)[i]
        mean.append(TimeSeriesList.from_times_and_values(time, m))

    mean = (TimeSeriesList(mean))
    lower_bounds = TimeSeriesList(lower_bounds)
    upper_bounds = TimeSeriesList(upper_bounds)

    # calculate intervals (from percentile) on transfer: 
    if series_transfer is not None:
        lower_bounds_transfer = []
        upper_bounds_transfer = []
        mean_transfer = []
        for i in range(bootstrap_predictions_transfer[0].len()): # iterate over different time series
            l_b=[]
            u_b = []
            time = bootstrap_predictions_transfer[0][i].time_index
            for a in alpha:
                lower = np.percentile(TimeSeriesList(bootstrap_predictions_transfer).values(), 100*(a/2), axis=0)[i] # .reshape(2,-1)
                upper = np.percentile(TimeSeriesList(bootstrap_predictions_transfer).values(), 100*(1-a/2), axis=0)[i]# .reshape(2,-1)
                l_b.append(TimeSeriesList.from_times_and_values(time, lower))
                u_b.append(TimeSeriesList.from_times_and_values(time, upper))
    
            lower_bounds_transfer.append(TimeSeriesList(l_b))
            upper_bounds_transfer.append(TimeSeriesList(u_b))
            m = np.percentile(TimeSeriesList(bootstrap_predictions_transfer).values(), 50, axis=0)[i]
            mean_transfer.append(TimeSeriesList.from_times_and_values(time, m))
    
        mean_transfer = (TimeSeriesList(mean_transfer))
        lower_bounds_transfer = TimeSeriesList(lower_bounds_transfer)
        upper_bounds_transfer = TimeSeriesList(upper_bounds_transfer)
    else:
        mean_transfer = None
        lower_bounds_transfer = None
        upper_bounds_transfer = None

    """
    if savingpath is not None:
        for i, n in enumerate(names):
            dataframe = pd.DataFrame({'time test': mean[i][0].time_index, 'mean test': mean[i][0].values().reshape(-1)})
            for j, a in enumerate(alpha):
                dataframe['lower '+str(a)] = lower_bounds[i][j][0].values().reshape(-1)
                dataframe['upper '+str(a)] = upper_bounds[i][j][0].values().reshape(-1)
            dataframe['time future'] = mean[i][1].time_index
            dataframe['mean future'] = mean[i][1].values().reshape(-1)
            for j, a in enumerate(alpha):
                dataframe['lower '+str(a)] = lower_bounds[i][j][1].values().reshape(-1)
                dataframe['upper '+str(a)] = upper_bounds[i][j][1].values().reshape(-1)

            dataframe.to_csv(savingpath + addition + n + '_bootstrapping_results(n='+str(n_bootstrap)+').txt', sep=';')

        for i, n in enumerate(names_transfer):
            print('transfer needs to be implemented')
            # TODO IMPLEMENT TRANSFER
    """

    return mean, lower_bounds, upper_bounds, mean_transfer, lower_bounds_transfer, upper_bounds_transfer




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

def block_bootstrap_(x, run_model, hyperparameters, n_bootstrap, n_pred, alphas=[0.05], x_transfer = None, savingpath=None):
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
    

