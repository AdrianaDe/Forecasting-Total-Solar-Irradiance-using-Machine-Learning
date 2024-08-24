from models import transformer
from loss_fn import PinballLoss
from tuning import backtesting
from evaluation import save_to_file
from evaluation import forecast_test_future
from bootstrapping import save_results
from load_data import load_and_prepare_data
from plotting import plot_with_interval
from plotting import plot_with_upper_and_lower

from plotting import plot_backtest

from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd

#####################################################
################# PARAMETERS ########################
#####################################################
names = ['reconstructed_tsi', 'ssn', 'phi', 'radio 10.7 cm'] #  ['reconstructed_tsi'] #
savingpath = '/cluster/home/adesassi/Final/Transformer/multi/' # !!! 

scaler = Scaler(StandardScaler())
start_year = 1968 #1968
n_pred = 11*12

# alphas = [0.05]
# n_bootstrap = 200 # !!!

# optimal hyperparameters
hyperparameters= ({'d_model': 368, 'n_head': 2, 'encoder_layers': 5, 'decoder_layers': 1, 'feedforward': 1024, 'dropout': 0.2, 'activation': 'relu', 'loss_function': PinballLoss(0.5), 'optimizer': torch.optim.Adam, 'learning_rate': 0.0005, 'seed': 0, 'batch_size': 256, 'max_epochs': 10, 'p_test': 2012, 'p_val': 0, 'n_in': 132, 'n_out': 12, 'smoothing': 12, 'outlier': 2.2})

# hyperparameters= ({'d_model': 2, 'n_head': 1, 'encoder_layers': 1, 'decoder_layers': 1, 'feedforward': 264, 'dropout': 0.2, 'activation': 'relu', 'loss_function': PinballLoss(0.5), 'optimizer': torch.optim.Adam, 'learning_rate': 0.01, 'seed': 0, 'batch_size': 256, 'max_epochs': 1, 'p_test': 2012, 'p_val': 0, 'n_in': 132, 'n_out': 12, 'smoothing': 12, 'outlier': 2.2})

                    
                    
#####################################################
################# LOAD DATA #########################
#####################################################
serie = load_and_prepare_data(names, split=hyperparameters['p_test'], scaler=scaler, outlier_threshold=hyperparameters['outlier'], smoothing_window=hyperparameters['smoothing'])


import matplotlib.pyplot as plt
plt.plot(serie[0].series.time_index, serie[0].series.values().reshape(-1), label='total')
plt.plot(serie[0].train.time_index, serie[0].train.values().reshape(-1), label='train')
plt.plot(serie[0].test.time_index, serie[0].test.values().reshape(-1), label='test')
plt.legend()
plt.savefig(savingpath + 'prepared_data.pdf')
plt.close()

plt.plot(serie[0].series_original.time_index, serie[0].series_original.values().reshape(-1))
plt.savefig(savingpath + 'original_data.pdf')
plt.close()


#####################################################
######## BACKTEST WTHOUT BOOTSTRAPPING ##############
#####################################################
print('\n BACKTEST')
historical_forecasts = backtesting(transformer, hyperparameters, names, start_year, n_pred, scaler=serie.scaler)
plot_backtest(serie, historical_forecasts, savingpath=savingpath)

print()
print('*'*30)
print('backtesting')
print('mse:', serie.mse(historical_forecasts))
print('*'*30)

save_to_file(historical_forecasts, names, savingpath, addition='backtest_')


#####################################################
######## FORCAST WITHOUT BOOTSTRAPPING ##############
#####################################################
model = transformer(train = serie.train, hyperparameters=hyperparameters)
prediction = forecast_test_future(serie, model, n_pred, scale_back=True)

plot_with_interval(serie, prediction, names=names, variance=None, reference_series=None, addition='', savingpath=savingpath)


print()
print('*'*30)
print('DIRECT FORECAST')
print('mse on test set:', serie.mse(prediction))
print('*'*30)

save_to_file(prediction, names, savingpath, addition='test_and_future_')


"""
#####################################################
################ BLOCK BOOTSTRAP ####################
#####################################################
# block bootstrapping gives the model uncertainty

mean, lower_bounds, upper_bounds, _, _, _ = block_bootstrap_(
    serie, transformer, hyperparameters, n_bootstrap, n_pred, alphas = alphas, x_transfer=None, savingpath=savingpath
)

plot_with_upper_and_lower(mean, prediction, upper_bounds, lower_bounds, reference_series = None, addition='', alphas = alphas, savingpath=savingpath)

save_results(prediction, mean, lower_bounds, upper_bounds, savingpath=savingpath, names=names, alpha=alphas, addition='')

print()
print('*'*30)
print('Block Bootstrap')
print('mse direct predict:', serie.mse(prediction))
print('mse mean predict:', serie.mse(mean))
print('*'*30)
"""