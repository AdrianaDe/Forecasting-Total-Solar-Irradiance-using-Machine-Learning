from darts.metrics import mse
from myTimeSeries import myTimeSeries
from myTimeSeries import TimeSeriesList
from darts import TimeSeries

import numpy as np
from copy import deepcopy
import pandas as pd

def get_error(series, prediction, scale_back=True): 
    '''
    PARAMETERS
    series (myTimeSeries or TimeSeriesList): data as myTimeSeries -> .series_original as reference
    prediction (TimeSEries or TimeSEriesList): corresponding predictions as TimeSeries (scaled)
    
    RETURNS
    list of mean squared errors of the predictions (compared to the original series, i.e. not prepared)
    '''
    if scale_back:
        prediction = series.scale_back(prediction)
    
    # 'series' is just a simple time series
    if isinstance(series, myTimeSeries): 
        # error of two simple time series
        if isinstance(prediction, myTimeSeries): 
            error = mse(series.series_original, prediction, intersect=True)
        # error of a simple time series and multiple predictions on it
        elif isinstance(prediction, TimeSeriesList): 
            error = []
            for pred in prediction: 
                error.append(mse(series.series_original, pred, intersect=True))
        else:
            print('error(0): type of prediction is not valid')
            sys.exit(1)
    # multiple series (eg. ssn, tsi,...)
    elif isinstance(series, TimeSeriesList):
        # for each series there is one corresponding prediction
        if isinstance(prediction[0], TimeSeries): 
            error = []
            for s, pred in zip(series, prediction): 
                error.append(mse(s.series_original, pred, intersect=np.True_))
        # for each series there are multiple prediction
        elif isinstance(prediction[0][0], TimeSeries):
            error = []
            for s, preds in zip(series, prediction):
                e = []
                for pred in preds:
                    e.append(mse(s.series_original, pred, intersect=True))
                error.append(e)
        else: 
            print('error(1): type of prediction is not valid')
            sys.exit(1)
    else:
        print('error(2): tpye of series is not valid')
        sys.exit(1)
    return error

    
def forecast_test_future(x, model, n_pred, scale_back = True):
    prediction = []
    prediction.append(model.predict(series = x.train, n=n_pred)) # prediction on test set
    prediction.append(model.predict(series=x.series, n = n_pred)) # prediction on future set

    # put prediction in right format
    prediction = list(map(list, zip(*prediction)))
    
    prediction = [TimeSeriesList(p) for p in prediction]
    if scale_back: 
        prediction = x.scale_back(TimeSeriesList(prediction))
    return TimeSeriesList(prediction)
    
    
def add_series(adding_series, var=False): 
    if var:
        p = 2
    else:
        p = 1

    tot = adding_series[0]**p
    for i in range(1, len(adding_series)):
        tot += adding_series[i]**p
        
    if var:
        return tot**(1/2)
    else:
        return tot


def add_interval(mean, lowers, uppers): 
    mean_tot = mean[0]
    for i in range(1, len(mean)):
        mean_tot += mean[i]

    lower_tot = []
    upper_tot = []
    for i in range(lowers[0].len()): # iterate over different series (tsi, ssn, ...)
        l_t = []
        u_t = []
        for a in range(lowers[0][0].len()): # iterate over different quantiles (alphas)
            l = (lowers[0][i][a] - mean[0][i])**2
            u = (uppers[0][i][a] - mean[0][i])**2
            for j in range(1, len(lowers)):
                l += (lowers[j][i][a] - mean[j][i])**2
                u += (uppers[j][i][a] - mean[j][i])**2
            l_t.append(TimeSeriesList(mean_tot[i] - l**(1/2)))
            u_t.append(TimeSeriesList(mean_tot[i] + u**(1/2)))
        lower_tot.append(TimeSeriesList(l_t))
        upper_tot.append(TimeSeriesList(u_t))

        return mean_tot, TimeSeriesList(lower_tot), TimeSeriesList(upper_tot)


def save_to_file(results, names, savingpath, addition=''):
    # save results to .txt file
    for i, n in enumerate(names):
        dataframe = pd.DataFrame({'time ' + str(0): results[i][0].time_index, 'value ' + str(0): results[i][0].values().reshape(-1)})
        for j in range(1, results[i].len()): 
            dataframe['time '+str(j)] = results[i][j].time_index
            dataframe['value '+str(j)] = results[i][j].values().reshape(-1)

        dataframe.to_csv(savingpath+addition+names[i] + '_prediction_results.txt', sep=';')

