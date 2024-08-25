from models import GRU
from load_data import load_data_list
from load_data import prepare_data
from bootstrapping import block_bootstrap_
from plotting import plot_with_upper_and_lower
from evaluation import forecast_test_future
from evaluation import add_series
from loss_fn import PinballLoss
from tuning import backtesting
from plotting import plot_backtest
from evaluation import add_interval
from bootstrapping import save_results

from vmdpy import VMD
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd

#####################################################
################# PARAMETERS ########################
#####################################################
# parameters for Variational Mode Decomposition
vmd_parameters = {
    "alpha": 1000,
    "tau": 0,
    "K": 20, # number of modes to be recovered >=1 (if K==1 -> no decomposition)
    "DC": 0,
    "init": 1,
    "tol": 1e-7,
}


names = ['reconstructed_tsi', 'tsi'] 
savingpath = './GRU/'
error_file = 'mse.txt'
scaler = Scaler(StandardScaler())
start_year = 1968
n_pred = 11*12
alpha = [0.05]  # Percentage outside of prediction interval
n_bootstrap = 200

# optimal hyperparameters
# optimal hyperparameters for IMF0
hyperparameters0 = {'hidden_dim': 25, 'n_rnn_layers': 2, 'dropout': 0.2, 'loss_function': PinballLoss(quantile=0.5), 
                     'likelihood': None, 'optimizer': torch.optim.Adam, 'learning_rate': 0.001, 'batch_size': 32, 
                     'max_epochs': 20, 'seed': 0, 'num_samples': 1, 'p_test': 2012, 'p_val': 0, 'n_in': 276, 'n_out': 1, 
                     'smoothing': None, 'outlier': None, 'weight_decay': 0} 

# optimal hyperparameters for IMF1
hyperparameters1 = {'hidden_dim': 12, 'n_rnn_layers': 2, 'dropout': 0.2, 'loss_function': PinballLoss(quantile=0.5),
                    'likelihood': None, 'optimizer': torch.optim.Adam, 'learning_rate': 0.001, 'batch_size': 32,
                    'max_epochs': 60, 'seed': 0, 'num_samples': 1, 'p_test': 2012, 'p_val': 0, 'n_in': 276, 'n_out': 6,
                    'smoothing': None, 'outlier': None, 'weight_decay': 0}
                    
                    
#####################################################
################# LOAD DATA #########################
#####################################################
data = load_data_list(names)

imf0 = []
imf1 = []
for d in data:
    u, _, _ = VMD(
        d['val'].values, vmd_parameters['alpha'], vmd_parameters['tau'], vmd_parameters['K'],
        vmd_parameters['DC'], vmd_parameters['init'], vmd_parameters['tol']
    )
    imf0.append(pd.DataFrame({'date': d['date'][:len(u[0])], 'val':u[0]}))
    imf1.append(pd.DataFrame({'date': d['date'][:len(u[1])], 'val':u[1]}))

# imf_names = ['reconstructed_tsi_imf0', 'tsi_imf0']
imf_names = ['reconstructed_tsi', 'tsi']

# IMF0 
imf0_reconstructed = (prepare_data([imf0[0]], [imf_names[0]], split=hyperparameters0['p_test'], scaler=Scaler(StandardScaler()), outlier_threshold=hyperparameters0['outlier'], smoothing_window=hyperparameters0['smoothing']))

imf0_tsi = (prepare_data([imf0[1]], [imf_names[1]], split=hyperparameters0['p_test'], scaler=imf0_reconstructed.scaler, outlier_threshold=hyperparameters0['outlier'], smoothing_window=hyperparameters0['smoothing']))

# IMF1
imf1_reconstructed = (prepare_data([imf1[0]], [imf_names[0]], split=hyperparameters1['p_test'], scaler=Scaler(StandardScaler()), outlier_threshold=hyperparameters1['outlier'], smoothing_window=hyperparameters1['smoothing']))
imf1_tsi = (prepare_data([imf1[1]], [imf_names[1]], split=hyperparameters1['p_test'], scaler=imf1_reconstructed.scaler, outlier_threshold=hyperparameters1['outlier'], smoothing_window=hyperparameters1['smoothing']))

# TOTAL DIRECT (observed data)
total_reconstructed = prepare_data([data[0]], ['reconstructed_tsi'], split=hyperparameters0['p_test'], scaler=Scaler(StandardScaler()), outlier_threshold=None, smoothing_window=None)
total_tsi = prepare_data([data[1]], ['tsi'], split=hyperparameters0['p_test'], scaler=total_reconstructed.scaler, outlier_threshold=None, smoothing_window=None)

# TOTAL IMF 0 + IMF 1
imf01_reconstructed = (prepare_data([pd.DataFrame({'date': imf0[0]['date'], 'val': imf0[0]['val'] + imf1[0]['val']})], [imf_names[0]], split=hyperparameters1['p_test'], scaler=None, outlier_threshold=None, smoothing_window=None))
imf01_tsi = (prepare_data([pd.DataFrame({'date': imf0[1]['date'], 'val': imf0[1]['val'] + imf1[1]['val']})], [imf_names[1]], split=hyperparameters1['p_test'], scaler=None, outlier_threshold=None, smoothing_window=None))


#####################################################
######## BACKTEST WTHOUT BOOTSTRAPPING ##############
#####################################################
print('\n BACKTEST IMF 0')
historical_forecasts0 = backtesting(GRU, hyperparameters0, ['reconstructed_tsi'], start_year, n_pred, scaler=imf0_reconstructed.scaler, data=[imf0[0]])
plot_backtest(imf0_reconstructed, historical_forecasts0, savingpath=savingpath+'imf0_', title='IMF 0')

print('\n BACKTEST IMF 1')
historical_forecasts1 = backtesting(GRU, hyperparameters1, ['reconstructed_tsi'], start_year, n_pred, scaler=imf1_reconstructed.scaler, data=[imf1[0]])
plot_backtest(imf1_reconstructed, historical_forecasts1, savingpath=savingpath+'imf1_', title='IMF 1')

# write error from backtesting into file
with open(savingpath+error_file, 'a') as file:
    print('\n', '*'*30, file=file)
    print('BACKTESTING ERROR', file=file)
    print('\n IMF0', file=file)
    print('mse direct predict:', imf0_reconstructed.mse(historical_forecasts0), file=file)
    print('\n IMF1', file = file)
    print('mse direct predict:', imf1_reconstructed.mse(historical_forecasts1), file=file)
    # print('*'*30)


print('\n BACKTEST IMF 0 + 1')
historical_forecasts01 = add_series([historical_forecasts0, historical_forecasts1], var = False)
plot_backtest(imf01_reconstructed, historical_forecasts01, savingpath=savingpath+'imf01_', title='IMF0 + IMF1', series_original = total_reconstructed)

# write error for final backtest into file
with open(savingpath+error_file, 'a') as file: 
    print('\n IMF0 + IMF1', file=file)
    print('mse compared to real imf 0 + imf1:', imf01_reconstructed.mse(historical_forecasts01), file=file)
    print('mse compared to observed data:', total_reconstructed.mse(historical_forecasts01), file=file)
    print('*'*30, file=file)


#####################################################
######## FORCAST WITHOUT BOOTSTRAPPING ##############
#####################################################

# IMF 0
model0 = GRU(train = imf0_reconstructed.train, hyperparameters=hyperparameters0)

prediction0 = forecast_test_future(imf0_reconstructed, model0, n_pred, scale_back=True)
prediction_transfer0 = forecast_test_future(imf0_tsi, model0, n_pred, scale_back=True)

plot_with_upper_and_lower(imf0_reconstructed, prediction0, None, None, reference_series = None, addition='IMF0 ', alphas = [], savingpath=savingpath+'direct_') 
plot_with_upper_and_lower(imf0_tsi, prediction_transfer0, None, None, reference_series = None, addition='IMF0 ', alphas = [], savingpath=savingpath+'direct_') 

# IMF 1
model1 = GRU(train = imf1_reconstructed.train, hyperparameters=hyperparameters1)

prediction1 = forecast_test_future(imf1_reconstructed, model1, n_pred, scale_back=True)
prediction_transfer1 = forecast_test_future(imf1_tsi, model1, n_pred, scale_back=True)


plot_with_upper_and_lower(imf1_reconstructed, prediction1, None, None, reference_series = None, addition='IMF1', alphas = [], savingpath=savingpath+'direct_') 
plot_with_upper_and_lower(imf1_tsi, prediction_transfer1, None, None, reference_series = None, addition='IMF1 ', alphas = [], savingpath=savingpath+'direct_') 



#####################################################
################ BLOCK BOOTSTRAP ####################
#####################################################

# IMF 0
mean0, lower_bounds0, upper_bounds0, mean_transfer0, lower_bounds_transfer0, upper_bounds_transfer0 = block_bootstrap_(
    imf0_reconstructed, GRU, hyperparameters0, n_bootstrap, n_pred, alphas=alpha, 
    x_transfer= imf0_tsi
)

plot_with_upper_and_lower(imf0_reconstructed, prediction0, upper_bounds0, lower_bounds0, reference_series = None, addition='IMF0 ', alphas = alpha, savingpath=savingpath) 
plot_with_upper_and_lower(imf0_tsi, prediction_transfer0, upper_bounds_transfer0, lower_bounds_transfer0, reference_series = None, addition='IMF0 ', alphas = alpha,  savingpath=savingpath)

save_results(prediction0, mean0, lower_bounds0, upper_bounds0, savingpath=savingpath, names=['reconstructed_tsi'], alpha=alpha, addition='IMF0_')
save_results(prediction_transfer0, mean_transfer0, lower_bounds_transfer0, upper_bounds_transfer0, savingpath=savingpath, names=['tsi'], alpha=alpha, addition='IMF0_')


# IMF 1
mean1, lower_bounds1, upper_bounds1, mean_transfer1, lower_bounds_transfer1, upper_bounds_transfer1 = block_bootstrap_(
    imf1_reconstructed, GRU, hyperparameters1, n_bootstrap, n_pred, alphas=alpha, 
    x_transfer=imf1_tsi
)

plot_with_upper_and_lower(imf1_reconstructed, prediction1, upper_bounds1, lower_bounds1, reference_series = None, addition='IMF1 ', alphas = alpha, savingpath=savingpath)
plot_with_upper_and_lower(imf1_tsi, prediction_transfer1, upper_bounds_transfer1, lower_bounds_transfer1, reference_series = None, addition='IMF1 ', alphas = alpha, savingpath=savingpath)

save_results(prediction1, mean1, lower_bounds1, upper_bounds1, savingpath=savingpath, names=['reconstructed_tsi'], alpha=alpha, addition='IMF1_')
save_results(prediction_transfer1, mean_transfer1, lower_bounds_transfer1, upper_bounds_transfer1, savingpath=savingpath, names=['tsi'], alpha=alpha, addition='IMF1_')

# write error into file
with open(savingpath+error_file, 'a') as file: 
    print('ERROR WITH BLOCK BOOTSTRAP', file=file)
    print('\n IMF0', file=file)
    print('NRL mse direct predict:', imf0_reconstructed.mse(prediction0), file=file)
    print('NRL mse mean predict:', imf0_reconstructed.mse(mean0), file=file)
    print('PMOD mse direct predict:', imf0_tsi.mse(prediction_transfer0), file=file)
    print('PMOD mse mean predict:', imf0_tsi.mse(mean_transfer0), file=file)

    print('\n IMF1', file=file)
    print('NRL mse direct predict:', imf1_reconstructed.mse(prediction1), file=file)
    print('NLR mse mean predict:', imf1_reconstructed.mse(mean1), file=file)
    print('PMOD mse direct predict:', imf1_tsi.mse(prediction_transfer1), file=file)
    print('PMOD mse mean predict:', imf1_tsi.mse(mean_transfer1), file=file)
    print('-'*30, file=file)



#####################################################
######## GET TOTAL: IMF0 + IMF1 #####################
#####################################################

mean01 = mean0 + mean1
mean_transfer01 = mean_transfer0 + mean_transfer1

prediction01, lower01, upper01 = add_interval([prediction0, prediction1], [lower_bounds0, lower_bounds1], [upper_bounds0, upper_bounds1])
prediction_transfer01, lower_transfer01, upper_transfer01 = add_interval([prediction_transfer0, prediction_transfer1], [lower_bounds_transfer0, lower_bounds_transfer1], [upper_bounds_transfer0, upper_bounds_transfer1])

plot_with_upper_and_lower(imf01_reconstructed, prediction01, upper01, lower01, reference_series = total_reconstructed, addition='IMF0 + IMF1 ', alphas = alpha, savingpath=savingpath) 
plot_with_upper_and_lower(imf01_tsi, prediction_transfer01, upper_transfer01, lower_transfer01, reference_series = total_tsi, addition='IMF0 + IMF1 ', alphas = alpha, savingpath=savingpath) 

save_results(prediction01, mean01, lower01, upper01, savingpath=savingpath, names=['reconstructed_tsi'], alpha=alpha, addition='IMF0_')
save_results(prediction_transfer01, mean_transfer01, lower_transfer01, upper_transfer01, savingpath=savingpath, names=['tsi'], alpha=alpha, addition='IMF0_')


# write error into file
with open(savingpath+error_file, 'a') as file: 
    print('\n IMF0 + IMF1', file=file)

    print('\n total compared to original', file=file)
    print('NRL mse direct predict:', total_reconstructed.mse(prediction01), file=file)
    print('NRL mse mean predict:', total_reconstructed.mse(mean01), file=file)
    print('PMOD mse direct predict:', total_tsi.mse(prediction_transfer01), file=file)
    print('PMOD mse mean predict:', total_tsi.mse(mean_transfer01), file=file)
    print('\n total compared to IMF0+IMF1', file=file)
    print('NRL direct predict:', imf01_reconstructed.mse(prediction01), file=file)
    print('NRL mean predict:', imf01_reconstructed.mse(mean01), file=file)
    print('PMOD direct predict:', imf01_tsi.mse(prediction_transfer01), file=file)
    print('PMOD mean predict:', imf01_tsi.mse(mean_transfer01), file=file)
    print('*'*30, file=file)

