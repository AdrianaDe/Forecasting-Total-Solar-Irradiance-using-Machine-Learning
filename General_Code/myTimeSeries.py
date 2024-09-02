from darts import TimeSeries
from copy import deepcopy
import pandas as pd
from datetime import datetime
from darts.utils.model_selection import train_test_split
import sys
from darts.metrics import mse
import numpy as np
import math

class myTimeSeries: 
    """
    myTimeSeries is designed to store and manage all essential information related to a time serie
    
    ATTRIBUTES: 
    series_original (darts.TimeSerie): contains the original (not prepared) time serie created from a pandas.DataFrame or darts.TimeSeries
    self.series (darts.TimeSerie): contains the same series but prepared (scaled, outlier removed, smoothed)
    self.strain (darts.TimeSeries): only training set of self.series
    self.test (darts.TimeSeries): only test set of self.series
    self.outlier_threshold (int): threshold used for outlier removement of the time series (None for no removal)
    self.smoothing_window (int): number of datapoints over which the the time series is averaged (None for no smoothing)
    self.scaler (darts Scaler): scaler which was used to scale self.series (None if no scaling is performed)
    """
    
    def __init__(self, data, split, scaler=None, outlier_threshold=None, smoothing_window=None, name='no name'):
        if isinstance(data, TimeSeries):
            self.series_original = data
        elif isinstance(data, pd.DataFrame):
            self.series_original = TimeSeries.from_dataframe(data, 'date', 'val')
        else: 
            print('series has no valid form, needs to be TimeSeries or pd.DataFrame')
            sys.exit(1)
        
        self.outlier_threshold = outlier_threshold
        self.scaler = deepcopy(scaler)
        self.smoothing_window = smoothing_window

        self.series = deepcopy(self.series_original)
        self.prepare()

        self.train, self.test = self.split_data(split)
        self.name = name

    def remove_outlier(self, window=12): 
        """
        remove outlier (i.e. datapoints which have a standard score larger than the threshold) from the prepared self.sereie by replacing them with a smoothed value

        PARAMETERS: 
        window (int): size of moving window, fixed number of observations used for smoothing
        threshold (float): threshold to decide if datapoint is outlier

        """
        if self.outlier_threshold == None:
            return
        # calculate standard score 
        data = deepcopy(self.series.pd_dataframe()).reset_index(drop=False)
        values = data['val']
        roll = values.rolling(window=window, min_periods=1, center=True)  # min_periods is minimum number of observations required in a window (at beginning/end)
        smoothed_values = roll.mean()
        std = roll.std()
        standard_score = (values-smoothed_values) / std

        # check each datapoint if outlier
        outlier = ~standard_score.between(-self.outlier_threshold, self.outlier_threshold) # true for outlier (outside threshold)

        # remove outliers from series
        for i in outlier[outlier == True].index:
            data.at[i, 'val'] = smoothed_values[i]
        
        self.series = TimeSeries.from_dataframe(data, 'date', 'val')
    
    def smooth(self): 
        """
        smoothes self.serie using a rolling mean with the window size self.smoothing_window
        """
        if self.smoothing_window is not None and self.smoothing_window > 0:
            df = self.series.pd_dataframe()
            df['val'] = df['val'].rolling(window=self.smoothing_window, center=True).mean()
            df = df.dropna()
            self.series = TimeSeries.from_dataframe(df)

    def scale(self):
        """
        scales self.serie using self.scaler. If scaler is already fitted it is used directly, otherwise it is fitted to self.sereie.  
        """
        if self.scaler is not None: 
            if not self.scaler._fit_called: # fit scaler if it was not fitted before
                self.scaler.fit(self.series)
            self.series =  (self.scaler).transform(self.series)  # scale the series
    
    def prepare(self): 
        self.remove_outlier()
        self.smooth()
        self.scale()

    def split_data(self, split): 
        if split is None or split == 0: # no split
            return self.series, None
        elif split > 1: # split at that year
            n = sum(self.series.time_index >= pd.Timestamp(split, 1, 1))
        else: # split is <=1 and therefore a percentage
            n = split

        return train_test_split(self.series, test_size=n)

    def scale_back(self, scaled_series): 
        """
        PARAMETER: 
        scaled_series (TimeSeries, TimeSeriesList): any time serie / list of time series. Usually the prediction of self.series
        
        RETURN: 
        (list of) TimeSeries that are inverse transformed with self.scaler
        """
        if self.scaler is not None: 
            if isinstance(scaled_series, TimeSeries):
                return self.scaler.inverse_transform(scaled_series)
            elif isinstance(scaled_series, TimeSeriesList):
                return TimeSeriesList([self.scale_back(s) for s in scaled_series.series_list])
            elif isinstance(scaled_series, list): 
                return [self.scale_back(s) for s in scaled_series]
            else: 
                print('ERROR: scaled_series has no valid format. Type is:')
                print(type(scaled_series))
                sys.exit(1)
        else: 
            return scaled_series

    def mse(self, other): 
        """
        returns the mean squared error between self.series_original and other
        
        other (myTimeSeries, TimeSeries or TimeSeriesList): for TimeSeriesList the mse is calculatet between self.series_original and each entry in other.series_list
        """
        if isinstance(other, myTimeSeries):
            return mse(self.series_original, other.series_original, intersect=True)
        elif isinstance(other, TimeSeries): 
            return mse(self.series_original, other, intersect=True)
        elif isinstance(other, TimeSeriesList):
            return [self.mse(o) for o in other]
        else:
            print('no valid fromat to calculate mse')
            sys.exit(1)
            
            
class TimeSeriesList:
    """
    Designed to make the handling of multiple time series easier. A
    
    ATTRIBUTES:
    self.series_list (list of darts TimeSeries or myTimeSeries): contains darts.TimeSeries, myTimeSeries or TimeSeriesList
    
    For TimeSeriesList any operation from myTimeSeries/TimeSeries can be used. These operations are performed element-wise. 
    EXAMPLE: 
    - TimeSeriesList.train returns a list which contains the training sets of all elements from TimeSerisList.series_list. 
    """
    def __init__(self, series_list):
        self.series_list = series_list
    
    def __getitem__(self, key):
        return self.series_list[key]
        
    def __setitem__(self, key, series):
        self.series_list[key] = series
        
    def __getattr__(self, name):
        """ 
        This is called if the attribut/method is not found in TimeSeriesList. 
        Instead it the attribute/method is applied on each entry of TimeSeriesList individual and a list of these results is returned.  applied to the entries of self.series_list and returns a list . 
        """
        attr = [getattr(series, name) for series in self.series_list]
        
        # if it is a method from e.g. myTimeSeries
        if callable(attr[0]):
            def method(*args, **kwargs):
                return [a(*args, **kwargs) for a in attr]
            return method
        # if it is an attribute (e.g. myTimeSeries.train)
        else:
            return attr
    
    def scale_back(self, scaled_series): 
        """
        For this self.series_list needs to contain myTimeSeries (not darts.TimeSeries). 
        uses self.series_list[i].scaler to scale back scaled_series[i] 
        
        scaled_series can have more dimensions than self.series_list. In this case, it is essential to ensure that the various scalers (from self) are applied along the first dimension. If the dimensions of scaled_series are arranged incorrectly, the wrong scalers will be applied to the wrong series. 
        
        PARAMETERS
        scaled_series (TimeSeriesList): time series that we want to scale back to original size
        
        RETURNS: 
        scaled back version of scaled_series (same dimension). 
        """
        series = []
        for i, x in enumerate(self.series_list):
            series.append(x.scale_back(scaled_series[i]))
        if isinstance(scaled_series, TimeSeriesList):
            return TimeSeriesList(series)
        else: 
            return series       
    
    def len(self):
        return len(self.series_list)
        
    def _add_and_sub(self, other, operation):
        """
        define + and - to be component wise
        self and other need to have same lenth and a list of darts.TimeSeries
        list of myTimeSeries can not be added because it is not meaningful to add differently scaled series 
        """

        if isinstance(other, TimeSeriesList):
            if other.len() == other.len(): 
                if operation == '+':
                    return TimeSeriesList([s + o for s, o in zip(self.series_list, other.series_list)])
                elif operation == '-':
                    return TimeSeriesList([s - o for s, o in zip(self.series_list, other.series_list)])
                else:
                    print('Error: operation ', operation, ' is not defined.')
                    sys.exit(1)
            else:
                print('Error(0): Only TimeSeriesList with same length can be added. The have length ', self.len(), ' and ', other.len())
                sys.exit(1)
        else:
            print('Error(1): Can only add a TimeSeriesList with other TimeSeriesList. Types are:')
            print(type(self), type(other))
            sys.exit(1)
    
    def __add__(self, other): 
        return self._add_and_sub(other, '+')

    def __sub__(self, other): 
        return self._add_and_sub(other, '-')
            
    def _mul_and_div(self, other, operation):
        """
        define + and - to be component wise
        
        PARAMETETERS
        other (TimeSeries List, int, float)
        operation: '*' or '/'
        """
        if isinstance(other, TimeSeriesList):
            if self.len() == other.len():
                if operation == '/':
                    return TimeSeriesList([s / o for s, o in zip(self.series_list, other.series_list)])
                elif operation == '*':
                    return TimeSeriesList([s * o for s, o in zip(self.series_list, other.series_list)])
                else:
                    print('Error: operation ', operation, ' is not defined.')
                    sys.exit(1)
            else: 
                 print('Error(0): Only TimeSeriesList with same length can be added. The have length ', self.len(), ' and ', other.len())
                 sys.exit(1)
        elif isinstance(other, (int, float)):
            if operation == '/':
                return TimeSeriesList([s / other for s in self.series_list])
            elif operation == '*':
                return TimeSeriesList([s * other for s in self.series_list])
            else:
                print('Error: operation ', operation, ' is not defined.')
                sys.exit(1)

    def __truediv__(self, other):
        return self._mul_and_div(other, '/')
    
    def __mul__(self, other):
        return self._mul_and_div(other, '*')
        
    def __rmul__(self, other):
        return self.__mul__(other)
        
    def __pow__(self, power):
        return TimeSeriesList([s**power for s in self.series_list])
    	
    def from_times_and_values(times, values): 
        """ 
        create a TimeSeriesList from a list of times and a list of values
        """
        list = []
        for i in range(len(times)):
            list.append(TimeSeries.from_times_and_values(times[i], values[i]))
        return TimeSeriesList(list)       
