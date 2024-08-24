from models import NaiveModel
from tuning import backtesting
from evaluation import save_to_file
from evaluation import forecast_test_future
from bootstrapping import save_results
from load_data import load_and_prepare_data
from plotting import plot_with_interval
from plotting import plot_with_upper_and_lower

from plotting import plot_backtest
from myTimeSeries import TimeSeriesList
from plotting import plot_with_upper_and_lower

from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd

#####################################################
################# PARAMETERS ########################
#####################################################
names = ['reconstructed_tsi'] #  ['reconstructed_tsi'] #
savingpath = '/cluster/home/adesassi/Final/Naive/' # !!! 

scaler = None
start_year = 1968 #1968
n_pred = 11*12

# optimal hyperparameters
hyperparameters= ({'smoothing': 53, 'outlier': None, 'n_in':1, 'n_out':1, 'k':132, 'p_test':2012})

                    
                    
#####################################################
################# LOAD DATA #########################
#####################################################
serie = load_and_prepare_data(names, split=hyperparameters['p_test'], scaler=scaler, outlier_threshold=hyperparameters['outlier'], smoothing_window=hyperparameters['smoothing'])


#####################################################
######## BACKTEST WTHOUT BOOTSTRAPPING ##############
#####################################################
print('\n BACKTEST')
historical_forecasts = backtesting(NaiveModel, hyperparameters, names, start_year, n_pred, scaler=serie.scaler)
plot_backtest(serie, historical_forecasts, savingpath=savingpath, series_original=serie)

print()
print('*'*30)
print('backtesting')
print('mse:', serie.mse(historical_forecasts))
print('*'*30)

save_to_file(historical_forecasts, names, savingpath, addition='backtest_')



#####################################################
######## FORCAST ON NRL ############################
#####################################################
serie = load_and_prepare_data(names, split=hyperparameters['p_test'], scaler=scaler, outlier_threshold=hyperparameters['outlier'], smoothing_window=hyperparameters['smoothing'])

model = NaiveModel(train = serie.train, hyperparameters=hyperparameters)
prediction = []
prediction.append(model.predict(n_pred)) # prediction on test set

serie = load_and_prepare_data(names, split=2024, scaler=scaler, outlier_threshold=hyperparameters['outlier'], smoothing_window=hyperparameters['smoothing'])

model = NaiveModel(train = serie.train, hyperparameters=hyperparameters)
prediction.append(model.predict(n_pred)) # future prediction

prediction = TimeSeriesList([TimeSeriesList(prediction)])

# plot_with_interval(serie, prediction, names=names, variance=None, reference_series=None, addition='', savingpath=savingpath)
plot_with_upper_and_lower(serie, prediction, None, None, reference_series = serie, alphas = [], savingpath=savingpath, addition='smoothed ')

print()
print('*'*30)
print('DIRECT FORECAST NRL')
print('mse on test set:', serie.mse(prediction))
print('*'*30)


save_to_file(prediction, names, savingpath, addition='test_and_future_')


#####################################################
######## FORCAST ON PMOD ############################
#####################################################
serie = load_and_prepare_data(['tsi'], split=hyperparameters['p_test'], scaler=scaler, outlier_threshold=hyperparameters['outlier'], smoothing_window=hyperparameters['smoothing'])

model = NaiveModel(train = serie.train, hyperparameters=hyperparameters)
prediction = []
prediction.append(model.predict(n_pred)) # prediction on test set

serie = load_and_prepare_data(['tsi'], split=2024, scaler=scaler, outlier_threshold=hyperparameters['outlier'], smoothing_window=hyperparameters['smoothing'])

model = NaiveModel(train = serie.train, hyperparameters=hyperparameters)
prediction.append(model.predict(n_pred)) # future prediction

prediction = TimeSeriesList([TimeSeriesList(prediction)])

# plot_with_interval(serie, prediction, names=names, variance=None, reference_series=None, addition='', savingpath=savingpath)
plot_with_upper_and_lower(serie, prediction, None, None, reference_series = serie, alphas = [], savingpath=savingpath, addition='smoothed ')

print()
print('*'*30)
print('DIRECT FORECAST PMOD')
print('mse on test set:', serie.mse(prediction))
print('*'*30)


save_to_file(prediction, ['tsi'], savingpath, addition='test_and_future_')
