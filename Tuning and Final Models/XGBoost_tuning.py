from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler
from load_data import load_and_prepare_data

from tuning import Choice
from tuning import Range
from tuning import Hyperparameters
from tuning import find_best_hyperparameters

from models import xgboost

#####################################################
################# PARAMETERS ########################
#####################################################
names = ['reconstructed_tsi'] #['reconstructed_tsi', 'ssn', 'phi', 'radio 10.7 cm'] #  

savingpath = './XGBoost/tuning/' 
scaler = Scaler(StandardScaler())
start_year = 1968
n_pred = 11*12
n_iterations = 1 # numer of random combination testet FOR EACH DATA PARAMETER COMBINATION (i.e. smoothing, outlier, n_in, n_out)

# Hyperparameter range over which we search best parameters
hyperparameters = Hyperparameters({
    "smoothing": Choice([None, 6, 12, 24]),
    "outlier": Choice([None, 2.2]),
    "p_val": 0,
    "p_test": 2012,
    "n_in": Choice([5*12, 7*12, 9*12, 6*12, 8*12, 11*12, 15*12, 19*12, 23*12]),
    "n_out": Choice([1, 6, 12, 24]),
    "encoders": None, 
    "seed": Choice([0]), 
    "max_depth": Choice([2, 3, 4, 5, 7, 9]), 
    "learning_rate": Choice([0.001, 0.01, 0.1, 0.5]), 
    "n_estimators": Choice([64, 128, 200, 256, 350, 512]), 
    "objective": Choice(['reg:squarederror', 'reg:quantileerror']),
    "quantile_alpha": Choice([[0.5]]), 
    "booster": Choice(['gbtree']), 
    "gamma": Choice([0]), 
    "reg_alpha": Choice([0, 0.05, 0.1, 0.5, 1, 10]), 
    "reg_lambda": Choice([0, 0.01, 0.05, 0.1, 1, 10]),
})

#####################################################
################# LOAD DATA #########################
#####################################################
series = load_and_prepare_data(names, split=hyperparameters.dic['p_test'], scaler=scaler, outlier_threshold=None, smoothing_window=None)

#####################################################
################# TUNING ############################
#####################################################
find_best_hyperparameters(hyperparameters, names, xgboost, start_year, n_pred, n_iterations, optimizing_i=0, savingpath=savingpath, scaler=series.scaler, method='data_all_model_random')
