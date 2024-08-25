from models import xgboost
from load_data import load_and_prepare_data
from bootstrapping import block_bootstrap_
from plotting import plot_with_upper_and_lower
from evaluation import forecast_test_future

from tuning import backtesting
from plotting import plot_backtest
from bootstrapping import save_results
from evaluation import get_error

from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler
import numpy as np

#####################################################
################# PARAMETERS ########################
#####################################################
savingpath = './XGBoost/'
scaler = Scaler(StandardScaler())
start_year = 1968
n_pred = 11*12
n_bootstrap = 200
error_file = 'mse.txt'

alphas = [0.05] 

names =  ['reconstructed_tsi'] # ['reconstructed_tsi', 'ssn', 'phi', 'radio 10.7 cm'] # 
transfer_names = ['tsi']  # ['tsi', 'ssn', 'phi', 'radio 10.7 cm'] # 

# PARAMETER SINGLE 
hyperparameters = {'encoders': None, 'seed': 0, 'max_depth': 9, 'learning_rate': 0.5, 'n_estimators': 350, 'objective': 'reg:quantileerror', 'quantile_alpha': [0.5], 'booster': 'gbtree', 'gamma': 0, 'reg_alpha': 1, 'reg_lambda': 0.05, 'p_test': 2012, 'p_val': 0, 'n_in': 96, 'n_out': 1, 'smoothing': None, 'outlier': 2.2}

# PARAMETER MULTI
# hyperparameters = {'smoothing': 12, 'outlier': None, 'p_val': 0, 'p_test': 2012, 'n_in': 72, 'n_out': 12, 'encoders': None, 'seed': 0, 'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 350, 'objective': 'reg:squarederror', 'quantile_alpha': [0.5], 'booster': 'gbtree', 'gamma': 0, 'reg_alpha': 0.05, 'reg_lambda': 0.05}


#####################################################
################# LOAD DATA #########################
#####################################################
series = load_and_prepare_data(names, split=hyperparameters['p_test'], scaler=scaler, outlier_threshold=None, smoothing_window=None)

series_transfer = load_and_prepare_data(transfer_names, split=hyperparameters['p_test'], scaler=scaler, outlier_threshold=None, smoothing_window=None)

"""
#####################################################
######## BACKTEST WTHOUT BOOTSTRAPPING ##############
#####################################################
historical_forecast = backtesting(xgboost, hyperparameters, names, start_year, n_pred, scaler=series.scaler)

plot_backtest(series, historical_forecast, savingpath=savingpath)
error = get_error(series, historical_forecast, scale_back=False)

# write error to error_file
with open(savingpath+error_file, 'w') as file: 
    print('\n' + '*'*30, file=file)
    print('BACKTESTING ERROR', file=file)
    for i in range(len(names)): 
        print('\n', names[i], ': ', error[i], file = file)
        print('mean: ', np.mean(error[0]), file = file)
    print('\n', file = file)
"""

#####################################################
######## FORCAST WITHOUT BOOTSTRAPPING ##############
#####################################################
# with NRL
model = xgboost(train=series.train, hyperparameters=hyperparameters)
prediction = forecast_test_future(series, model, n_pred, scale_back=True)

# with PMOD
model_transfer = xgboost(train=series_transfer.train, hyperparameters=hyperparameters)
prediction_transfer = forecast_test_future(series_transfer, model_transfer, n_pred, scale_back=True)

# plot results
plot_with_upper_and_lower(series, prediction, None, None, reference_series=series, alphas=[], savingpath=savingpath+'direct_')
plot_with_upper_and_lower(series_transfer, prediction_transfer, None, None, reference_series=series_transfer, alphas=[], savingpath=savingpath+'direct_')


#####################################################
################ BLOCK BOOTSTRAP ####################
#####################################################
# block bootstrapping with NRL and PMOD
mean, lower_bounds, upper_bounds, _, _, _ = block_bootstrap_(series, xgboost, hyperparameters, n_bootstrap, n_pred, alphas=alphas, x_transfer=None, savingpath=savingpath)
mean_transf, lower_bounds_transf, upper_bounds_transf, _, _, _ = block_bootstrap_(series_transfer, xgboost, hyperparameters, n_bootstrap, n_pred, alphas=alphas, x_transfer=None, savingpath=savingpath)

# plot results from bootstrapping
plot_with_upper_and_lower(series, prediction, upper_bounds, lower_bounds, reference_series=None, alphas=alphas, savingpath=savingpath)
plot_with_upper_and_lower(series_transfer, prediction_transfer, upper_bounds_transf, lower_bounds_transf, reference_series=None, alphas=alphas, savingpath=savingpath)

# print to error_file
with open(savingpath+error_file, 'w') as file: 
    print('\n' + '*'*30, file=file)
    print('RESULTS', file=file)
    print('\n NRL', file=file)
    for i in range(len(names)): 
        print(names[i], ' direct: ', series[i].mse(prediction[i]), file=file)
        print('mean: ',  series[i].mse(mean[i]), file=file)
    print('\n PMOD', file=file)
    for i in range(len(names)):
        print('mse direct predict: ', series_transfer[i].mse(prediction_transfer[i]), file=file)
        print('mse mean predict:', series_transfer[i].mse(mean_transf[i]), file=file)
    print('*'*30, file=file)

# save the predictions to a .txt file
save_results(prediction, mean, lower_bounds, upper_bounds, savingpath=savingpath, names=names, alpha=alphas, addition='')
save_results(prediction_transfer, mean_transf, lower_bounds_transf, upper_bounds_transf, savingpath=savingpath, names=['tsi'], alpha=alphas, addition='')

