from models import NaiveModel
from tuning import backtesting
from evaluation import save_to_file
from evaluation import get_error
from load_data import load_and_prepare_data
from plotting import plot_with_upper_and_lower

from plotting import plot_backtest
from myTimeSeries import TimeSeriesList

import numpy as np

#####################################################
################# PARAMETERS ########################
#####################################################
names = ['reconstructed_tsi'] #  ['reconstructed_tsi'] #
savingpath = './Naive/' 
error_file = 'mse.txt'

scaler = None
start_year = 1968 
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
historical_forecasts = backtesting(NaiveModel, hyperparameters, names, start_year, n_pred, scaler=serie.scaler)
plot_backtest(serie, historical_forecasts, savingpath=savingpath, series_original=serie)


# save historical forecasts into a txt file 
save_to_file(historical_forecasts, names, savingpath, addition='backtest_')

# error = serie.mse(historical_forecasts)
error = get_error(serie, historical_forecasts, scale_back=False)
# write error into a txt file
with open(savingpath+error_file, 'a') as file: 
    print('\n' + '*'*30, file=file)
    print('BACKTESTING ERROR', file=file)
    for i in range(len(names)): 
        print('\n', names[i], ': ', error[i], file = file)
        print('mean: ', np.mean(error[i]), file = file)
    print('\n', file = file)


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

plot_with_upper_and_lower(serie, prediction, None, None, reference_series = serie, alphas = [], savingpath=savingpath, addition='smoothed ')


error = serie.mse(prediction)

# write error to error_file
with open(savingpath+error_file, 'a') as file: 
    print('\n', file=file)
    print('DIRECT FORECAST NRL mean square error on test set', file=file)
    for i in range(len(names)): 
        print('\n', names[i], ': ', error[i], file = file)
    print('\n', file = file)


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

plot_with_upper_and_lower(serie, prediction, None, None, reference_series = serie, alphas = [], savingpath=savingpath, addition='smoothed ')

error = serie.mse(prediction)
# write error to error_file
with open(savingpath+error_file, 'a') as file: 
    print('DIRECT FORECAST PMOD mean square error on test set', file=file)
    for i in range(len(names)): 
        print('\n', names[i], ': ', error[i], file = file)
    print('\n', file = file)
    print('*'*30, file=file)


save_to_file(prediction, ['tsi'], savingpath, addition='test_and_future_')
