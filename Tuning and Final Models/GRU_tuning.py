from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler
from vmdpy import VMD
import torch
import pandas as pd

from load_data import load_and_prepare_data
from load_data import load_data_list
from load_data import prepare_data
from loss_fn import PinballLoss

from tuning import Choice
from tuning import Range
from tuning import Hyperparameters
from tuning import find_best_hyperparameters
from models import GRU

#####################################################
################# PARAMETERS ########################
#####################################################
names = ['reconstructed_tsi']
savingpath = './GRU/tuning/' 
scaler = Scaler(StandardScaler())
start_year = 1968
n_pred = 11*12
n_iterations = 100 # numer of random combination testet FOR EACH DATA PARAMETER COMBINATION (i.e. smoothing, outlier, n_in, n_out)

hyperparameters = Hyperparameters({
    # "early_stopping_patience": Choice([7]),
    "smoothing": Choice([None]),
    "outlier": Choice([None]),
    "p_val": 0,
    "p_test": 2012,
    "n_in": Choice([11*12, 15*12, 20*12, 23*12, 25*12]),
    "n_out": Choice([1, 4, 6, 8, 12]), 
    "hidden_dim": Choice([8, 12, 16, 25, 50]), 
    "n_rnn_layers": Choice([1, 2, 4]), 
    "dropout": Choice([0, 0.1, 0.2, 0.3]),
    "loss_function": Choice([PinballLoss(quantile=0.5), torch.nn.MSELoss()]), 
    "likelihood": Choice([None]),
    "optimizer": Choice([torch.optim.Adam]),
    "learning_rate": Choice([1e-3, 1e-2]),
    "batch_size": Choice([16, 32, 64, 128]),
    "max_epochs": Choice([20, 40, 60]),
    "seed": Choice([0]),
    "num_samples": Choice([1]), 
})

# parameters for Variational Mode Decomposition
vmd_parameters = {
    "alpha": 1000,
    "tau": 0,
    "K": 20, # number of modes to be recovered >=1 (if K==1 -> no decomposition)
    "DC": 0,
    "init": 1,
    "tol": 1e-7,
}

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
    

#####################################################
################# TUNING ############################
#####################################################
find_best_hyperparameters(hyperparameters, names, GRU, start_year, n_pred, n_iterations, optimizing_i=0, savingpath=savingpath+'imf0/', scaler=scaler, method='data_all_model_random', data=imf0)

find_best_hyperparameters(hyperparameters, names, GRU, start_year, n_pred, n_iterations, optimizing_i=0, savingpath=savingpath+'imf1/', scaler=scaler, method='data_all_model_random', data=imf1)

