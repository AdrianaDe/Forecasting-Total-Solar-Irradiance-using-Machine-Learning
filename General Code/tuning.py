import random
import itertools
import numpy as np
import pandas as pd

from datetime import datetime
from dateutil.relativedelta import relativedelta
from load_data import load_and_prepare_data
from load_data import prepare_data
from evaluation import get_error

from myTimeSeries import myTimeSeries
from myTimeSeries import TimeSeriesList

class Range: # Hyperparameter can be in the range between (l, u)
    def __init__(self, l, u):
        self.lower = l
        self.upper = u

    def get_random(self):
        if (type(self.lower) == int) and (type(self.upper) == int):
            return random.randint(self.lower, self.upper)
        else:
            return random.uniform(self.lower, self.upper)

class Choice: # Hyperparameter can be one value out of a list
    def __init__(self, choices):
        self.choices = choices

    def get_random(self):
        return random.choice(self.choices)

    def get_all(self):
        return self.choices
        
 
class Hyperparameters:
    def __init__(self, dic):
        self.dic = dic

    def get_random_combination(self):
        new_dict = {}
        for i in self.dic:
            if isinstance(self.dic[i], Choice) or isinstance(self.dic[i], Range):
                new_dict[i] = self.dic[i].get_random()
            else:
                new_dict[i] = self.dic[i]
        return new_dict

    def get_all_combinations(self):
        keys = list(self.dic.keys())
        values = []
        for k in keys:
            if isinstance(self.dic[k], Choice):
                values.append(self.dic[k].get_all())
            elif isinstance(self.dic[k], Range):
                print('Error: can not get all out of a range')
                sys.exit(1)
            else:
                values.append([self.dic[k]])

        all_combinations_tuples = list(itertools.product(*values))
        return [dict(zip(keys, combination)) for combination in all_combinations_tuples]
      
    def get_all_data_combinations(self): 
        data_parameters = ["p_test", "p_val", "n_in", "n_out", "smoothing", "outlier"]
        return Hyperparameters({key: self.dic[key] for key in data_parameters}).get_all_combinations()
        
        
        
def backtesting(run_model, hyperparameters, names, start_year, n_pred, stride=None, start_month = 1, start_day=1, scaler=None, data=None):
    """
    Get on historical data forecasts with distance 'stride', the model is retrained on data before for each forecast.

    PARAMETERS:
    run_model (function from models.py): function to generate and train a model
    hyperparameters (Hyperparameters): contains all hyperparameters of the model
    names (list of str): list with names of the time series on which we want to train and backtest
    star_year (int): year where first historical forecast begins
    n_pred (int): length of one forecast (in number of data points)
    stride (int): distance between two forecasts (in number of data points)
    scaler (Scaler): how to scale data before training

    RETURNS:
    historical_forecasts (list of TimeSeries): list of all historical forecasts.
    -> scaled back to the original size
    -> the first entry of the list is a list with all forecasts of the first dataset
    """
    if stride is None:
        stride = n_pred

    # only for probabilistic Darts models num_samples > 1
    num_samples = 1
    if "num_samples" in hyperparameters:
        num_samples = hyperparameters['num_samples']

    # length of the shift (stride) and prediction, only for monthly data
    diff_stride = relativedelta(months=stride)
    diff_pred = relativedelta(months=n_pred)

    year = datetime(start_year, start_month, start_day)

    historical_forecast = []
    while (year + diff_pred) <= datetime(hyperparameters["p_test"], 1, 1):
        print('backtesting at ', year, '...')
        if data is None:
            x = load_and_prepare_data(names, split=year.year, scaler=scaler, outlier_threshold=hyperparameters['outlier'], smoothing_window=hyperparameters['smoothing'])
        else:
            x = prepare_data(data, names, split=year.year, scaler=scaler, outlier_threshold=hyperparameters['outlier'], smoothing_window=hyperparameters['smoothing'])
        model = run_model(hyperparameters, x.train)
        try:
            predictions = model.predict(n_pred, series=x.train, num_samples=num_samples)
        except TypeError:  # This is only for the naive models
            predictions = [model.predict(n_pred, num_samples=num_samples)]
        historical_forecast.append(x.scale_back(predictions))

        year += diff_stride
    
    # transpose such that in first dimension the different series (e.g. tsi, ssn) and in second dimenstion the different backcast of each series
    historical_forecast = list(map(list, zip(*historical_forecast)))

    return TimeSeriesList([TimeSeriesList(hf) for hf in historical_forecast])
    
    
    
def find_best_hyperparameters(hyperparameters, names, run_model, start_year, n_pred, n_iterations=1, optimizing_i=0, savingpath=None, scaler=None, model_number=0, method='random'):
    """
    method (str): 'all': go over all parameters, 'random': completely random over all hyperparameters, 'data_all_model_random': look in each dataparameter for random
    """
    if method == 'all': 
        parameters = hyperparameters.get_all_combinations()
    elif method == 'random': 
        parameters = []
        for i in range(n_iterations):
            parameters.append(hyperparameters.get_random_combination())
    elif method == 'data_all_model_random':
        best_error = np.inf
        best_hyperparameters = None
        best_historical_forecast = None
        data_parameters = hyperparameters.get_all_data_combinations()
        for p in data_parameters:
            model_number += 1
            new_hyperparameters = {key: value for key, value in hyperparameters.dic.items() if key not in data_parameters}
            new_hyperparameters.update(p)
            b_e, b_h, b_hf = find_best_hyperparameters(Hyperparameters(new_hyperparameters), names, run_model, start_year, n_pred, n_iterations, optimizing_i, savingpath, scaler, model_number, method='random')
            if b_e < best_error:
                best_error = b_e
                best_hyperparameters = b_h
                best_historical_forecast = b_hf
        return best_error, best_hyperparameters, best_historical_forecast

    x = load_and_prepare_data(names, split=hyperparameters.dic['p_test'], scaler=scaler, outlier_threshold=None, smoothing_window=None)
    
    best_error = np.inf
    best_hyperparameters = None
    best_historical_forecast = None
    
    for j, p in enumerate(parameters): 
        hf = backtesting(run_model, p, names, start_year, n_pred, scaler=x.scaler)
        error = get_error(x, hf, scale_back=False)
        if np.mean(error[optimizing_i]) < best_error: 
            best_error = np.mean(error[optimizing_i])
            best_hyperparameters = p
            best_historical_forecast = hf
        
        if savingpath is not None: 
            with open(savingpath + 'all_results.txt', 'a') as file: 
                print('*'*30, 'combination ', str(j), file=file)
                print('\n', 'Hyperparameters:', file=file)
                print(p, file=file) # print current hyperparameters
                for i in range(len(error)):
                    print('\n error of '+ names[i], file=file)
                    print(error[i], file=file)
                print('', file=file)
    
    if savingpath is not None:
        with open(savingpath + 'best_models_results.txt', 'a') as file:
            print('*'*30, ' model ', str(model_number), file=file)
            print('\n', 'best hyperparameters: ', file=file)
            print(best_hyperparameters, file=file)
            print(' ', file=file)
            print(names[optimizing_i] + ' mean squared error over validation set: ' + str(best_error), file=file)

    # x = load_and_prepare_data(names, split=best_hyperparameters['p_test'], scaler=scaler, outlier_threshold=best_hyperparameters['outlier'], smoothing_window=best_hyperparameters['smoothing'])

    return best_error, best_hyperparameters, best_historical_forecast # , x
    
    
def save_backtest(historical_forecasts, names, savingpath, addition=''):
    # save results to .txt file
    for j, n in enumerate(names): 
        for i, hf in enumerate(historical_forecasts[j]):
            if i == 0:
                dataframe = pd.DataFrame({'time '+str(i): hf.time_index, 'value '+str(i): hf.values().reshape(-1)})
            else:
                dataframe['time '+str(i)] = hf.time_index
                dataframe['value '+str(i)] = hf.values().reshape(-1)
        
        dataframe.to_csv(savingpath + addition + names[j] + '_HistoricalForecasts.txt', sep=';')
        
        
