import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

from datetime import datetime
import sys

# colorblind friendly color scheme :) 
mycolors = ['#000000',  '#fc8d59', '#d73027', '#4575b4', '#91bfdb', '#e0f3f8', '#fee090'] 

def get_standard_figure(a = 13, b=5, fontsize=18):
    """
    prepares a figure with standard format to plot later

    PARAMETERS:
    a (int), b (int): figsize is (a,b)
    fontsize (int)

    RETURNS:
    ax: ax of the figure to plot things afterwards
    """
    plt.rcParams.update({'font.size': fontsize, 'legend.frameon': True, 'legend.framealpha': 0.5, 'legend.facecolor': 'white', 'lines.linewidth':2})
    
    fig, ax = plt.subplots(figsize=(a, b))
    ax.grid(True, linestyle='-', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_xlabel('Year')
    
    return ax

def get_plot_parameters(name):
    """
    returns for each type of TimeSeries define the ylabel for a plot and label

    PARAMETERS:
    name (str): name of data type

    RETURNS:
    label, ylabel
    """
    if name=='tsi' or name=='PMOD':
        return 'PMOD', 'Total Solar Irradiance [W/m\u00B2]'

    elif name=='reconstructed_tsi' or name=='tsi_reconstructed' or name=='NRL':
        return 'NRL', 'Total Solar Irradiance [W/m\u00B2]'

    elif name=='ssn':
        return 'SILSO', 'Sunspot Number'

    elif name=='phi':
        return 'Usokin', 'Modulation Potential $\Phi$ [MV]'

    elif name=='radio 10.7 cm' or name=='F10.7':
        return 'CLS', 'Absolute Solar Flux \nat 10.7 cm [SFU]'

    else: # any not defined names have no ylabel and no label
        return ' ', ' '
        

def plot_backtest(series, historical_forecast, indices = None, savingpath=None, x_begin=1960, ylim=None, title=None, series_original = None):
    """
    Generate a plot of historical forecasts
    
    PARAMETERS:
    series (TimeSeriesList of myTimeSeries): series on which forecasts are done
    historical_forecast (TimeSeriesList of dart.TimeSeries): historical forecasts
    indices (list of int): If only specific series should be plot. None -> plot all
    savingpath (str): where to save plot. None -> is not saved, only plt.show()
    x_begin (int): year where to cut the plot on the left side (right side is always end of prediction) 
    xlim (list of float): for cutting plot on y-axis (e.g. [0, 53.5])
    title (str): title of plot
    series_original (TimeSeriesList of myTimeSeries):  the Time Series without any preparation to plot in background as reference (gray)
    """
    if indices is None:
        indices = range(historical_forecast.len()) 
    elif type(indices) == int:
        indices = [indices]

    for i in indices: 
        ax = get_standard_figure()
        label, ylabel = get_plot_parameters(series[i].name)
        if series_original is not None:
            series_original[i].series_original.plot(color=mycolors[0], label=label, linewidth=2, alpha=0.1)
        series[i].scale_back(series[i].series).plot(color=mycolors[0], label='prepared '+label, linewidth=2)
        for j in range(historical_forecast[i].len()):
            # series[i].scale_back(historical_forecast[i][j]).plot( color=mycolors[j%(len(mycolors)-1) + 1], label='historical forecast ' + str(historical_forecast[i][j].start_time().year), linewidth=2)
            (historical_forecast[i][j]).plot( color=mycolors[j%(len(mycolors)-1) + 1], label='hist. forecast ' + str(historical_forecast[i][j].start_time().year), linewidth=2)
        ax.set_ylabel(ylabel)
        ax.set_label('year')
        ax.set_xlim(datetime(x_begin, 1,1), historical_forecast[i][-1].end_time())
        ax.legend(ncol=3)

        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        if title is not None:
            plt.title(title)
        if savingpath is not None:
            plt.savefig(savingpath + 'backtest_'+str(series[i].name)+'.pdf', bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        
def plot_with_upper_and_lower(x, means, uppers=None, lowers=None, reference_series = None, addition='', alphas = [], savingpath=None, x_start=1995, ylim=None):
    """
    Generate a plot with optionally prediction interval
    
    PARAMETERS:
    x (TimeSeriesList of myTimeSeries): series used for training and we want to predict on
    mean (TimeSeriesList of darts.TimeSeries): mean of forecast (dark line in plot). Includes forecast on test set and forecast in future
    uppers, lowers (TimeSeriesList of darts.TimeSeries): upper and lower limit for forecast
    reference_series (myTimeSeries or None): the Time Series without any preparation to plot in background (gray)
    addition (str): some additional label when saving figures
    alphas (list of floats): list of interval numbers (e.g. [0.05, 0.1] means there are two prediction intervals, one for 95% and one for 90%) -> [] for no intervals
    savintpath (str): path to save plot. If None -> not saved only plt.show()
    x_start (int): where to cut the plot on the left side (right side is always end of prediction) 
    y_lim (list of float): for cutting plot on y-axis (e.g. [0, 53.5])
    """
    test_color = mycolors[2]
    future_color = mycolors[4]
    for i in range(means.len()):
        ax = get_standard_figure()
        label, ylabel = get_plot_parameters(x[i].name)
        if reference_series is not None:
            reference_series[i].series_original.plot(label=label, ax=ax, alpha=0.2)

        x[i].scale_back(x[i].series).plot(label=addition+label, ax=ax, color=mycolors[0])
        means[i][0].plot(label='forecast ' + str(means[i][0].time_index[0].year), ax=ax, color=test_color)
        means[i][1].plot(label='forecast ' + str(means[i][1].time_index[0].year), ax=ax, color=future_color)

        for j,a in enumerate(alphas):
            ax.fill_between(uppers[i][j][0].time_index, lowers[i][j][0].values().reshape(-1), uppers[i][j][0].values().reshape(-1), color=test_color, alpha=0.2+0.1*i, label=str(100*(1-a))+ '% prediction interval')
            ax.fill_between(uppers[i][j][1].time_index, lowers[i][j][1].values().reshape(-1), uppers[i][j][1].values().reshape(-1), color=future_color, alpha=0.2+0.1*j, label=str(100*(1-a))+ '% prediction interval')

        ax.legend(ncol=3)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('year')
        ax.set_xlim(datetime(x_start, 1, 1), means.time_index[i][1][-1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        if savingpath is not None:
            plt.savefig(savingpath+addition+label+'_prediction.pdf', bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        
 
        

