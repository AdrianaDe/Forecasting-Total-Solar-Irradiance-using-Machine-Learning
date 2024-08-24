from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import EarlyStopping
from darts.models.forecasting.transformer_model import TransformerModel
from darts.models import RNNModel
from darts.models import BlockRNNModel
from darts.models import XGBModel
import torch
from sklearn.metrics import mean_squared_error
from darts.models import NaiveSeasonal

        
def xgboost(hyperparameters, train): 
    """
    PARAMETERS:
    series (List with myTimeSeries): data on which we want to train the model
    train (list of TimeSeries): training data
    
    RETURNS:
    model (Darts Model): trained XGBoost-Model
    """
    
    if hyperparameters["objective"] == 'reg:quantileerror':
        model = XGBModel( lags= list(range(-hyperparameters["n_in"], 0)), output_chunk_length= hyperparameters["n_out"], add_encoders= hyperparameters["encoders"], random_state= hyperparameters["seed"], multi_models= False,  max_depth= hyperparameters["max_depth"], learning_rate= hyperparameters["learning_rate"], n_estimators= hyperparameters["n_estimators"], objective= hyperparameters["objective"], booster= hyperparameters["booster"], gamma = hyperparameters["gamma"], reg_alpha =  hyperparameters["reg_alpha"], quantile_alpha=hyperparameters['quantile_alpha'])
    else:
        model = XGBModel(
            lags= list(range(-hyperparameters["n_in"], 0)),
            output_chunk_length= hyperparameters["n_out"],
            add_encoders= hyperparameters["encoders"],
            random_state= hyperparameters["seed"],
            multi_models= False,  # False means it trains one models for multiple time series, 
            # pl_trainer_kwargs = {"callbacks": [loss_logger, early_stopping]},
            # lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
            # lr_scheduler_kwargs = {"mode": 'min', "factor": hyperparameters["reduce_lr_factor"], "patience": hyperparameters["reduce_lr_patience"], "min_lr": 1e-10},
            max_depth= hyperparameters["max_depth"],
            learning_rate= hyperparameters["learning_rate"],
            n_estimators= hyperparameters["n_estimators"],
            objective= hyperparameters["objective"],
            booster= hyperparameters["booster"],
            gamma = hyperparameters["gamma"],
            reg_alpha =  hyperparameters["reg_alpha"])

    # train model
    model.fit(series=train, verbose=True)

    return model  
    

def transformer(hyperparameters, train, verbose=False): 
    """
    PARAMETERS:
    hyperparameter (dict): all hyperparameters needed for Transformer
    train (List with myTimeSeries): data on which we want to train the model
    verbose (bool): plot some additional information (e.g. progress) during training 
    
    RETURNS:
    model (Darts Model): trained Transformer-Model
    """
    model = TransformerModel(
        input_chunk_length = hyperparameters["n_in"],
        output_chunk_length = hyperparameters["n_out"],
        output_chunk_shift = 0,
        d_model = hyperparameters["d_model"] , # needs to be a multiple of nhead
        nhead = hyperparameters["n_head"],
        num_encoder_layers=hyperparameters["encoder_layers"],
        num_decoder_layers=hyperparameters["decoder_layers"],
        dim_feedforward=hyperparameters["feedforward"],
        dropout = hyperparameters["dropout"],
        activation=hyperparameters["activation"],  # ['GLU', 'Bilinear', 'ReGLU', 'GEGLU', 'SwiGLU', 'ReLU', 'GELU', 'relu', 'gelu']
        custom_encoder=None,
        custom_decoder=None,
        loss_fn = hyperparameters["loss_function"], 
        optimizer_cls = hyperparameters["optimizer"],
        optimizer_kwargs={'lr': hyperparameters["learning_rate"]},
        # lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
        # lr_scheduler_kwargs = {"mode": 'min', "factor": hyperparameters["reduce_lr_factor"], "patience": hyperparameters["reduce_lr_patience"], "min_lr": 1e-10},
        # pl_trainer_kwargs = {"callbacks": [loss_logger, early_stopping]},
        random_state=hyperparameters["seed"], 
        batch_size = hyperparameters["batch_size"],
        n_epochs = hyperparameters["max_epochs"],
    )

    model.fit(series=train, verbose = verbose)

    return model


def RRN(hyperparameters, train, model_type, verbose=False):
    """
    Generates and trains LSTM/GRU model

    PARAMETERS:
    hyperparameter (dict): all hyperparameters needed for LSTM/GRU
    train (List with myTimeSeries): data on which we want to train the model
    model_type: either 'LSTM' or 'GRU'

    RETURNS:
    model (Darts Model): trained LSTM/GRU-Model
    """

    model = BlockRNNModel(
        model=model_type,
        input_chunk_length = hyperparameters["n_in"],
        output_chunk_length = hyperparameters["n_out"],
        hidden_dim= hyperparameters["hidden_dim"],
        n_rnn_layers = hyperparameters["n_rnn_layers"],
        dropout = hyperparameters["dropout"],
        loss_fn = hyperparameters["loss_function"],  # ignored if likelihood != None
        likelihood = hyperparameters["likelihood"],
        optimizer_cls = hyperparameters["optimizer"],
        optimizer_kwargs={'lr': hyperparameters["learning_rate"], 'weight_decay': hyperparameters['weight_decay']},
        # lr_scheduler?
        batch_size=hyperparameters["batch_size"],
        n_epochs= hyperparameters["max_epochs"],
        random_state = hyperparameters["seed"],
        # training_length = hyperparameters["n_in"] + hyperparameters["n_out"],  # length of input and output time series used during training
    )

    # train model
    model.fit(series=train, verbose=verbose)

    return model

def GRU(hyperparameters, train, verbose=False):
    return RRN(hyperparameters, train, 'GRU', verbose=verbose)

def LSTM(hyperparameters, train, verbose=False):
    return RRN(hyperparameters, train, 'LSTM', verbose=verbose)


def NaiveModel(hyperparameters, train, i=0):
    """
    Generate Naive Seasonal Model (i.e. model that repeats the value after k steps)

    PARAMETERS:
    hyperparameters (dict): only hyperparameter k 
    train (List with myTimeSeries): data on which we want to train the model
    i (int): Naive model can only train on series. i is the index of the series we train (repeat the series)

    RETURNS:
    trained darts model
    """
    model = NaiveSeasonal(K=hyperparameters["k"])
    model.fit(train[i])
    return model

