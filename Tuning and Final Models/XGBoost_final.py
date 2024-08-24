from models import xgboost
from load_data import load_and_prepare_data
from load_data import load_data_list
from load_data import prepare_data

from bootstrapping import get_all_blocks
from bootstrapping import resample_with_replacement
from bootstrapping import add_residual
from bootstrapping import mean_and_variance
from bootstrapping import block_bootstrap_

from bootstrapping import resuidual_bootstrapping
from plotting import plot_with_upper_and_lower

from evaluation import forecast_test_future
from evaluation import add_series
from plotting import plot_with_interval

from loss_fn import PinballLoss
from tuning import backtesting
from plotting import plot_backtest
from evaluation import add_interval
from bootstrapping import save_results
from evaluation import get_error

from vmdpy import VMD
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd
import numpy as np

from bootstrapping import save_results


#####################################################
################# PARAMETERS ########################
#####################################################
savingpath = '/cluster/home/adesassi/Final/XGBoost/single/'  # !!!
scaler = Scaler(StandardScaler())
start_year = 1968
n_pred = 11*12
n_bootstrap = 200

alphas = [0.05] 

names =  ['reconstructed_tsi'] # ['reconstructed_tsi', 'ssn', 'phi', 'radio 10.7 cm'] # 

# best parameter only on reconstructed_tsi after tuning: 
# hyperparameters = {'encoders': None, 'seed': 0, 'max_depth': 4, 'learning_rate': 0.1, 'n_estimators': 200, 'objective': 'reg:squarederror', 'quantile_alpha': [0.5], 'booster': 'gbtree', 'gamma': 0, 'reg_alpha': 0.05, 'reg_lambda': 0.01, 'p_test': 2012, 'p_val': 0, 'n_in': 72, 'n_out': 1, 'smoothing': 24, 'outlier': 1}

# one of the best hyperparameters on reconstructed_tsi: (model 8: 0.07057524362409658; model 54: 0.07050159644109734)
# hyperparameters = {'encoders': None, 'seed': 0, 'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 200, 'objective': 'reg:squarederror', 'quantile_alpha': [0.5], 'booster': 'gbtree', 'gamma': 0, 'reg_alpha': 0.1, 'reg_lambda': 1, 'p_test': 2012, 'p_val': 0, 'n_in': 60, 'n_out': 1, 'smoothing': 12, 'outlier': 2.2}
# hyperparameters = {'encoders': None, 'seed': 0, 'max_depth': 9, 'learning_rate': 0.1, 'n_estimators': 256, 'objective': 'reg:squarederror', 'quantile_alpha': [0.5], 'booster': 'gbtree', 'gamma': 0, 'reg_alpha': 0.1, 'reg_lambda': 10, 'p_test': 2012, 'p_val': 0, 'n_in': 72, 'n_out': 1, 'smoothing': 12, 'outlier': None}

# not best hyperparameter, but shows good cycle (tuned only on reconstructed tsi), model 146
hyperparameters = {'encoders': None, 'seed': 0, 'max_depth': 9, 'learning_rate': 0.5, 'n_estimators': 350, 'objective': 'reg:quantileerror', 'quantile_alpha': [0.5], 'booster': 'gbtree', 'gamma': 0, 'reg_alpha': 1, 'reg_lambda': 0.05, 'p_test': 2012, 'p_val': 0, 'n_in': 96, 'n_out': 1, 'smoothing': None, 'outlier': 2.2}

# best hyperparameters when tuning on all datasets
# hyperparameters = {'smoothing': 12, 'outlier': 2.2, 'p_val': 0, 'p_test': 2012, 'n_in': 72, 'n_out': 12, 'encoders': None, 'seed': 0, 'max_depth': 7, 'learning_rate': 0.1, 'n_estimators': 512, 'objective': 'reg:squarederror', 'quantile_alpha': [0.5], 'booster': 'gbtree', 'gamma': 0, 'reg_alpha': 0.1, 'reg_lambda': 0.01}

# PARAMETER MULTI
# hyperparameters = {'smoothing': 12, 'outlier': None, 'p_val': 0, 'p_test': 2012, 'n_in': 72, 'n_out': 12, 'encoders': None, 'seed': 0, 'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 350, 'objective': 'reg:squarederror', 'quantile_alpha': [0.5], 'booster': 'gbtree', 'gamma': 0, 'reg_alpha': 0.05, 'reg_lambda': 0.05}


#####################################################
################# LOAD DATA #########################
#####################################################
series = load_and_prepare_data(names, split=hyperparameters['p_test'], scaler=scaler, outlier_threshold=None, smoothing_window=None)

series_transfer = load_and_prepare_data(['tsi'], split=hyperparameters['p_test'], scaler=series.scaler, outlier_threshold=None, smoothing_window=None)

#####################################################
######## BACKTEST WTHOUT BOOTSTRAPPING ##############
#####################################################
historical_forecast = backtesting(xgboost, hyperparameters, names, start_year, n_pred, scaler=series.scaler) # Scaler(StandardScaler())

plot_backtest(series, historical_forecast, savingpath=savingpath)
error = get_error(series, historical_forecast, scale_back=False)

print()
print('*'*30)
print('BACKTESTING ERROR')
print(names)
print(error)
print('*'*30)
print(error[0])
print(np.mean(error[0]))
print('-'*30)


#####################################################
######## FORCAST WITHOUT BOOTSTRAPPING ##############
#####################################################
model = xgboost(train=series.train, hyperparameters=hyperparameters)
prediction = forecast_test_future(series, model, n_pred, scale_back=True)
prediction_transfer = forecast_test_future(series_transfer, model, n_pred, scale_back=True)

plot_with_interval(series, prediction, names, variance=None, reference_series=None, addition='', savingpath=savingpath+'direct_')
plot_with_interval(series_transfer, prediction_transfer, ['tsi'], variance=None, reference_series=None, addition='', savingpath=savingpath+'direct_')

#####################################################
################ BLOCK BOOTSTRAP ####################
#####################################################
mean, lower_bounds, upper_bounds, mean_transfer, lower_bounds_transfer, upper_bounds_transfer = block_bootstrap_(series, xgboost, hyperparameters, n_bootstrap, n_pred, alphas=alphas, x_transfer=series_transfer, savingpath=savingpath)

plot_with_upper_and_lower(series, prediction, upper_bounds, lower_bounds, reference_series=None, alphas=alphas, savingpath=savingpath)
plot_with_upper_and_lower(series_transfer, prediction_transfer, upper_bounds_transfer, lower_bounds_transfer, reference_series=None, alphas=alphas, savingpath=savingpath)

print()
print('*'*39)
for i in range(len(names)): 
    print('mse direct predict:', series[i].mse(prediction[i]))
    print('mse mean predict:', series[i].mse(mean[i]))
    print()
print('transfer ', series_transfer.name)
print('mse direct predict:', series_transfer.mse(prediction_transfer))
print('mse mean predict:', series_transfer.mse(mean_transfer))
print('*'*30)


save_results(prediction, mean, lower_bounds, upper_bounds, savingpath=savingpath, names=names, alpha=alphas, addition='')
save_results(prediction_transfer, mean_transfer, lower_bounds_transfer, upper_bounds_transfer, savingpath=savingpath, names=['tsi'], alpha=alphas, addition='')

#####################################################
###### BLOCK BOOTSTRAP trained with PMOD ############
#####################################################
names = ['tsi'] # ['tsi', 'ssn', 'phi', 'radio 10.7 cm'] 
series = load_and_prepare_data(names = names, split=hyperparameters['p_test'], scaler=scaler, outlier_threshold=None, smoothing_window=None)

model = xgboost(train=series.train, hyperparameters=hyperparameters)
prediction = forecast_test_future(series, model, n_pred, scale_back=True)
plot_with_interval(series, prediction, names, variance=None, reference_series=None, addition='', savingpath=savingpath+'final_direct_')

mean, lower_bounds, upper_bounds, mean_transfer, lower_bounds_transfer, upper_bounds_transfer = block_bootstrap_(series, xgboost, hyperparameters, n_bootstrap, n_pred, alphas=alphas, x_transfer=None, savingpath=savingpath+'final_')
plot_with_upper_and_lower(series, prediction, upper_bounds, lower_bounds, reference_series=None, alphas=alphas, savingpath=savingpath+'final_')
save_results(prediction, mean, lower_bounds, upper_bounds, savingpath=savingpath+'final_', names=names, alpha=alphas, addition='')

print()
print('*'*30)
print('results trained with tsi')
for i in range(len(names)): 
    print('mse direct predict:', series[i].mse(prediction[i]))
    print('mse mean predict:', series[i].mse(mean[i]))
    print()
print('*'*30)


