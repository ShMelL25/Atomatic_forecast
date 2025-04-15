import optuna
from optuna.samplers import TPESampler
import statsmodels.api as sm
import itertools
from sklearn.metrics import mean_squared_error,mean_absolute_error
from .models import Model
import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.stattools import adfuller, kpss
warnings.filterwarnings("ignore")

class Optimizer:

    def __init__(self, data:np.array):
        """
        Initializes the Forecaster object.

        Parameters:
        data (np.array): Time series data used for forecasting.

        Attributes:
        self.data (np.array): Time series data.
        self.pdq (list of tuples): List of tuples (p, d, q) representing possible combinations of parameters for the ARIMA model.
        self.pdqs (list of tuples): List of tuples (p, d, q, 12) representing possible combinations of parameters for the seasonal ARIMA model.

        Methods:
        self.acf_pacf_(data): Method to compute the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) of the data.
        self.stationarity_check(data): Method to check the stationarity of the data.

        Note:
        - self.pdq and self.pdqs are generated based on the ACF and PACF of the data, as well as the stationarity check.
        - The seasonal parameter for the ARIMA model is set to 12, which corresponds to monthly data.
        """
        self.data = data
        q, p = self.acf_pacf_(self.data)
        d = [self.stationarity_check(self.data)]
        self.pdq = list(itertools.product(list(p), d, list(q)))
        self.pdqs = list(itertools.product(list(p), d, list(q), [12]))
        

    def opt_run(self, n_trials:int):
        """
        Runs the optimization process for SARIMAX and Exponential Smoothing models.

        Parameters:
        n_trials (int): Number of trials to run for the optimization process.

        Returns:
        tuple: A tuple containing the best parameters for the SARIMAX model and the Exponential Smoothing model.

        Description:
        This method creates and optimizes two separate studies using Optuna for the SARIMAX and Exponential Smoothing models.
        It uses the `objective_sarimax` and `objective_es` methods to define the optimization objectives for each model.
        The optimization process is run for the specified number of trials (`n_trials`).

        The method returns the best parameters found for both models.
        """
        study_sarimax=optuna.create_study(direction="minimize", sampler=TPESampler())
        study_sarimax.optimize(self.objective_sarimax,n_trials=n_trials, n_jobs=-1)
        
        study_es=optuna.create_study(direction="minimize", sampler=TPESampler())
        study_es.optimize(self.objective_es,n_trials=n_trials, n_jobs=-1)
        
        best_sarimax_params = study_sarimax.best_params
        best_sarimax_score = study_sarimax.best_value
        best_es_params = study_es.best_params
        best_es_score = study_es.best_value
        
        # Фильтрация
        if best_sarimax_score != float('inf'):
            best_sarimax_result = best_sarimax_params
        else:
            best_sarimax_result = None
        
        if best_es_score != float('inf'):
            best_es_result = best_es_params
        else:
            best_es_result = None

        return best_sarimax_result, best_es_result
        

    def objective_sarimax(self, trial:optuna.Trial):
        """
        Objective function for optimizing the SARIMAX model using Optuna.

        Parameters:
        trial (optuna.Trial): A trial object from Optuna, used to suggest hyperparameters.

        Returns:
        float: The Root Mean Squared Error (RMSE) of the model's predictions, which serves as the objective value to minimize.

        Description:
        This method defines the objective function for optimizing the SARIMAX model. It uses Optuna's trial object to suggest
        hyperparameters for the model, including the order, seasonal order, and trend. These hyperparameters are then used
        to create a SARIMAX model and make predictions.

        The method calculates the Root Mean Squared Error (RMSE) between the actual data and the model's predictions,
        which is returned as the objective value to be minimized by Optuna.
        """
        order=trial.suggest_categorical('order',self.pdq)
        seasonal_order=trial.suggest_categorical('seasonal_order',self.pdqs)
        trend=trial.suggest_categorical('trend',['n','c','t','ct',None])
        sarima_kwargs = {
            'order':order, 
            'seasonal_order':seasonal_order, 
            'trend':trend
            }
        try:
            model = Model(x=self.data[:-12], sarimax_kwargs=sarima_kwargs, es_kwargs=None) 
            predictions = model.predict_arima(11)
            rmse = mean_squared_error(self.data, predictions)
            accuracy=rmse
            for i in predictions:
                if i < 0:
                    return np.inf
            return accuracy
        except np.linalg.LinAlgError:
            return np.inf
    
    def objective_es(self, trial:optuna.Trial):
        """
        Objective function for optimizing the Exponential Smoothing model using Optuna.

        Parameters:
        trial (optuna.Trial): A trial object from Optuna, used to suggest hyperparameters.

        Returns:
        float: The Root Mean Squared Error (RMSE) of the model's predictions, which serves as the objective value to minimize.

        Description:
        This method defines the objective function for optimizing the Exponential Smoothing model. It uses Optuna's trial object to suggest
        hyperparameters for the model, including the seasonal type, trend type, initialization method, use of Box-Cox transformation,
        and seasonal periods. These hyperparameters are then used to create an Exponential Smoothing model and make predictions.

        The method calculates the Root Mean Squared Error (RMSE) between the actual data and the model's predictions,
        which is returned as the objective value to be minimized by Optuna.
        """
        
        seasonal=trial.suggest_categorical('seasonal',['add', 'additive', 'multiplicative'])
        trend=trial.suggest_categorical('trend',['add', 'additive', 'multiplicative'])
        initialization_method=trial.suggest_categorical('initialization_method',['heuristic','estimated'])
        use_boxcox=trial.suggest_categorical('use_boxcox',[True,False])
        seasonal_periods=trial.suggest_categorical('seasonal_periods',[12])
        es_kwargs={
            'seasonal':seasonal, 
            'trend':trend, 
            'seasonal_periods':seasonal_periods, 
            'use_boxcox':use_boxcox,
            'initialization_method':initialization_method}

        model = Model(x=self.data[:-12], sarimax_kwargs=None, es_kwargs=es_kwargs) 
        predictions = model.predict_es(11)
        
        for i in predictions:
            if i < 0:
                return np.inf
        rmse = mean_squared_error(self.data, predictions)
        accuracy=rmse
        return accuracy
    
    def stationarity_check(self, data:np.array)->int:
        """
        Checks the stationarity of the time series data using the Augmented Dickey-Fuller (ADF) test.

        Parameters:
        data (np.array): The time series data to be checked for stationarity.

        Returns:
        int: The number of differences required to make the time series stationary.

        Description:
        This method iteratively applies differencing to the time series data and performs the ADF test to check for stationarity.
        It starts with the original data and applies up to 10 levels of differencing. For each level of differencing, it performs
        the ADF test and prints the p-value. If the p-value is less than or equal to 0.05, the method concludes that the time
        series is stationary and returns the number of differences applied.

        Note:
        - The method uses the `adfuller` function from the `statsmodels` library to perform the ADF test.
        - The method prints a message indicating whether the time series is stationary based on the ADF test results.
        """
        for i in range(10):
            if i > 0:
                data = np.diff(data, n=i)
            
            result_adf = adfuller(data)
            if result_adf[1] <= 0.05:
                return i
        
    def acf_pacf_(self, data, lags:int=10, n:int=3):
        """
        Computes the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) of the time series data.

        Parameters:
        data (np.array or pd.Series): The time series data for which ACF and PACF are to be computed.
        lags (int): The number of lags for which ACF and PACF are to be computed. Default is 10.
        n (int): The number of minimum values to search for in the ACF and PACF arrays. Default is 3.

        Returns:
        tuple: A tuple containing the minimum values from the ACF and PACF arrays.

        Description:
        This method calculates the ACF and PACF of the time series data for a specified number of lags.
        It then searches for the minimum values in the ACF and PACF arrays using the `search_min` method.
        The method returns the minimum values from the ACF and PACF arrays, which can be used to determine
        the order of the ARIMA model.

        Note:
        - The method uses the `acf` and `pacf` functions from the `statsmodels.tsa.stattools` module to compute ACF and PACF.
        - The `search_min` method is used to find the minimum values in the ACF and PACF arrays.
        """
        lag_acf = sm.tsa.stattools.acf(data, nlags=lags)  
        lag_pacf = sm.tsa.stattools.pacf(data, nlags=lags)  
        
        lag_acf_min = self.search_min(arr=lag_acf, n=n)
        lag_pacf_min = self.search_min(arr=lag_pacf, n=n)

        return lag_acf_min, lag_pacf_min
    
    def search_min(self, arr:np.array, n:int)->np.array:
        """
        Searches for the indices of the smallest `n` values in the absolute value array.

        Parameters:
        arr (np.array): The array in which to search for the smallest values.
        n (int): The number of smallest values to find.

        Returns:
        np.array: An array containing the indices of the smallest `n` values in the absolute value array.

        Description:
        This method computes the absolute values of the input array and then finds the indices of the smallest `n` values
        in the absolute value array. It uses the `np.argpartition` function to efficiently find the indices of the smallest
        values without fully sorting the array.

        Note:
        - The method uses the `np.abs` function to compute the absolute values of the input array.
        - The `np.argpartition` function is used to find the indices of the smallest `n` values.
        """
        abs_arr = np.abs(arr)
        indices = np.argpartition(abs_arr, n)[:n]
        
        return indices