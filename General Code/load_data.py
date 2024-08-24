from myTimeSeries import myTimeSeries
from myTimeSeries import TimeSeriesList

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import math

# path = '/content/drive/MyDrive/Colab Notebooks/DATA/'  # path on google Colab
path = '/cluster/home/adesassi/DATA/'  # path on euler


def from_JulianDay_to_year(julian_day, start_year = 1721424.5): 
    """
    Converts a list of Julian Day numbers to Gregorian dates.

    Parameters:
    julian_day (list): A list of Julian Day numbers to be converted.
    start_year (float): The Julian Day number corresponding to the start of the calendar (default georgian dates)

    Returns:
    list: A list of datetime objects representing the corresponding Gregorian dates.
    """
    date = [datetime.fromordinal(math.floor(julian_day[i] - start_year)) for i in range(julian_day.shape[0])]
    return date
    
def to_monthly(df, name='val', day_of_month=1):
    """
    converts a pd.DateFrame with daily data into a pd.DateFrame with monthly data.
    A monthly datapoint is the mean of all values during this month

    Parameters:
    df (pandas DataFrame): data with time entry 'date' and values saved in the entry name
    name (str): name where the values are saved
    day_of_month (int): at which date the value is saved (default: 1, at beginning of month)
    
    Returns:
    pandas DataFrame: monthly Data
    """
    years = np.array([date.year for date in df['date']])
    months = np.array([date.month for date in df['date']])
    days = np.array([date.day for date in df['date']])
    
    monthly_value = []
    monthly_date = []
    for y in np.arange(years[0], years[-1]+1, 1):
        # data in this year
        this_df = df[(df['date'] >= datetime(y, 1, 1)) & (df['date'] < datetime(y+1, 1, 1))]
        this_months = months[years == y] 

        # go through all month in this year (usually 12, but not at beginning and end)
        for m in np.arange(this_months[0], this_months[-1]+1, 1): 
            monthly_date.append(datetime(year=y, month=m, day=day_of_month))
            monthly_value.append(np.mean(df[(years==y) & (months == m)][name]))

    df_m = pd.DataFrame({'val': monthly_value, 'date': monthly_date})  # 'year_dec': [d.year + d.month/12 for d in monthly_date]

    return df_m

def decimal_year_to_date(decimal_year):
    year = int(decimal_year)
    rest = decimal_year - year
    
    # Determine if the year is a leap year
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        days_in_year = 366
    else:
        days_in_year = 365
    
    # Calculate the number of days corresponding to the rest
    days = rest * days_in_year
    start_date = datetime(year, 1, 1)
    
    # Calculate the exact date
    exact_date = start_date + timedelta(days=int(days))
    
    return exact_date
    

def import_tsi(freq = 'm'):
    '''
    freq = 'd', 'm' or 'y' for daily, monthly or yearly data
    '''
    names = ['Time_dec', 'Time_JD', 'TSI', 'Unc', 'TSI_after_corr', 'Unc_after_corr']
    tsi = pd.read_csv(path + "TSI_MergedPMOD_NobaselineScaleCycle23_JPM_Dec2023.txt", delimiter=' ', names=names, skiprows=1)
    
    # add year, month and day (converted from JD date)
    tsi['date'] = from_JulianDay_to_year(tsi['Time_JD'])
    
    if freq == 'm':
        tsi = to_monthly(tsi, 'TSI_after_corr')

    elif freq == 'd':
        tsi = tsi
        # tsi = pd.DataFrame({'val': tsi['TSI_after_corr'].values, 'year_dec': tsi['Time_dec'].values, 'date': tsi['date']})
        
    elif freq == 'y':
        print('yearl not implemented, TODO')
        return 0
   
    else: 
        print(freq, 'is not a valid frequency. Choose daily \'d\', monthly \'m\' or yearly \'y\'')
        return 0
        
    return tsi
    
    
def import_reconstructed_tsi(freq='m'):
    if freq == 'd':
        print('No dayli recontructed data available. Use import_tsi().')
        
    elif freq=='m': 
        tsi = pd.read_csv(path + 'nrl2_tsi_P1M.csv', delimiter=',') 
        d = datetime.toordinal(datetime(1610,1,1))
        date = from_JulianDay_to_year(tsi['time (days since 1610-01-01)'], -d)
        tsi = pd.DataFrame({'date': date, 'val': tsi['irradiance (W/m^2)'].values, 'unc': tsi['uncertainty (W/m^2)']})
        tsi = to_monthly(tsi, 'val')
        return tsi
    
    elif freq=='y':
        tsi = pd.read_csv(path + 'historical_tsi.csv', delimiter=',')
        date = [datetime(y, 1, 1) for y in tsi['time (yyyy)']]
        tsi = pd.DataFrame({'date': date, 'val': tsi['Irradiance (W/m^2)'].values, 'Time_dec': tsi['time (yyyy)']})
        return tsi
        
    else:
        print(freq, 'is not a valid frequency. Choose daily \'d\', monthly \'m\' or yearly \'y\'')

    return 0


    
def import_ssn(freq = 'm'):
    '''
    freq = 'd', 'm' or 'y' for daily, monthly or yearly data
    '''
    if freq == 'm':
        names = ['year', 'month', 'year_dec', 'val', 'std', 'number_obs', '?']
        ssn = pd.read_csv(path + "SN_m_tot_V2.0.csv", delimiter=';', names=names)
        date_ssn = np.array([datetime(year=ssn['year'][i], month=ssn['month'][i], day=1) for i in range(ssn.shape[0])])
        ssn['date'] = date_ssn
        
        ssn = ssn.loc[:, ('val', 'year_dec', 'date')]
    
    elif freq == 'd': 
        print('daily not implemented, TODO')
        return 0
    elif freq == 'y': 
        print('yearly not implemented, TODO')
        return 0
    else: 
        print(freq, 'is not a valid frequency. Choose daily \'d\', monthly \'m\' or yearly \'y\'')
        return 0
    
    return ssn
    
    
def import_mg2(freq = 'm'):
    '''
    freq = 'd', 'm' or 'y' for daily, monthly or yearly data
    '''
    if freq == 'm':
        names = ['fractional_year', 'month', 'day', 'index', 'unc'] # 'fractional_year',
        mg = pd.read_csv(path + "MgII_composite_edited.csv", delimiter=';', names=names)
        mg = mg[mg['fractional_year']<=2023.95]
        
        # convert daily MGII data to monthly MGII data
        mg_monthly = []
        mg_monthly_date = []
        for y in np.arange(int(mg['fractional_year'][0]), int(mg['fractional_year'][mg.shape[0]-1])+1, 1):
            this_year = mg[(mg['fractional_year'] >= y) & (mg['fractional_year'] < y+1)]
            for m in np.arange(this_year['month'][this_year.index[0]], this_year['month'][this_year.index[-1]]+1, 1):
                mg_monthly_date.append(datetime(year=y, month=m, day=1))
                mg_monthly.append(np.mean(this_year[this_year['month']==m]['index']))

        
        mg = pd.DataFrame({'val': mg_monthly, 'year_dec': [d.year + d.month/12 for d in mg_monthly_date], 'date': mg_monthly_date})
    
    elif freq == 'd': 
        names = ['fractional_year', 'month', 'day', 'index', 'unc'] # 'fractional_year',
        mg = pd.read_csv(path + "MgII_composite_edited.csv", delimiter=';', names=names)
        mg = mg[mg['fractional_year']<=2023.95]
        
        date = [datetime(math.floor(mg['fractional_year'].values[i]), mg['month'].values[i], mg['day'].values[i]) for i in mg.index]
        mg = pd.DataFrame({'val': mg['index'], 'year_dec': mg['fractional_year'], 'date': date})
        
    elif freq == 'y': 
        print('yearly not implemented, TODO')
        return 0
    else: 
        print(freq, 'is not a valid frequency. Choose daily \'d\', monthly \'m\' or yearly \'y\'')
        return 0
    
    return mg


def import_phi(freq='m'):
    phi = pd.read_csv(path + "Phi_monthly_1951-2023(utf-8).csv", delimiter=',')
    if freq == 'd':
        print('no daily data for phi available')
        return 0
    elif freq == 'm':
        date = [decimal_year_to_date(dec_year) for dec_year in phi['Year']]
        date_range = pd.date_range(start='1951-02-01', end='2023-12-01', freq='MS')
        date_array = np.array(date_range.to_pydatetime())
        phi = pd.DataFrame({'date': date_array, 'val': phi['Phi_(MV)'], 'Time_dec': phi['Year'], 'date_original': date})
        return phi
    elif freq =='y':
        print('TODO: Implement frequency y for phi')
        return 0
    else:
        print('Frequency ' + str(freq) + ' is not valid. Choose yearly (y), monthly (m) or daily (d).')

def import_radio107(freq = 'm', name='107'):
    data = pd.read_csv(path + 'cls_radio_flux_f'+name+'.csv', delimiter=',', skiprows=0) 
    date = []
    for d in data['time (yyyy MM dd)']:
        if int(int(d[5:7]) < 10):
            date.append(datetime(int(d[0:4]), int(d[5:6]), int(d[7:9])))
        else:
            date.append(datetime(int(d[0:4]), int(d[5:7]), int(d[8:10])))
    
    data = pd.DataFrame({'date': date, 'val': data['absolute_f'+name+'_c (solar flux unit (SFU))'].values})
    if freq == 'd':
        return data
    elif freq == 'm': 
        return to_monthly(data)
    elif freq == 'y':
        print('TODO: implement importing yearly radio10.7 & radio15 data.')
        return 0


def import_radio150(freq='m'): 
    return import_radio107(freq=freq, name='15')


def get_import_function(name):
    if name=='tsi' or name == 'PMOD':
        return import_tsi
    elif name=='reconstructed_tsi' or name=='long_tsi' or name=='NRL' or name=='tsi_reconstructed':
        return import_reconstructed_tsi
    elif name=='ssn':
        return import_ssn
    elif name=='phi': 
        return import_phi
    elif name=='radio 10.7 cm' or name=='radio107' or name=='radio 107 mm':
        return import_radio107
    elif name=='radio 15 cm' or name=='radio150' or name=='radio 150 mm': 
        return import_radio150
    elif name=='mg2':
        return import_mg2
    else:
        print(name, ' is no valid name!')
        return 0


def load_data_list(names, freq='m'): 
    """
    PARAMETER
    name (list of str): names of the data we want to load
    
    RETURN
    pd.DataFrame of the data. At least column 'date' with timestapms and column 'val' with values
    """
    data = []
    for n in names: 
        data.append(get_import_function(n)(freq))

    return data
    
def prepare_data(data, names, split, scaler=None, outlier_threshold=None, smoothing_window=None):
    x = []
    for i, d in enumerate(data):
        if isinstance(scaler, list):
            sc = scaler[i]
        else:
            sc = scaler
        x.append(myTimeSeries(d, split = split, scaler=sc, outlier_threshold=outlier_threshold, smoothing_window=smoothing_window, name=names[i]))
        
    return TimeSeriesList(x)
    

def load_and_prepare_data(names, split, scaler=None, outlier_threshold=None, smoothing_window=None):
    """
    Load a list of data and fully prepare it.
    
    PARAMETER
    name (list of str): names of the data we want to load and prepare
    split (int): year where to split data into train/validation & test set
    outlier_threshold (int): threshold for outlier removal
    smoothing window (int): size_of_window for smoothing
    
    RETURN
    TimeSeriesList with the myTimeSeries corresponding to names
    """
    data = load_data_list(names)
    
    return prepare_data(data, names, split, scaler, outlier_threshold, smoothing_window)
