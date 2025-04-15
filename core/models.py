import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class Model:

    def __init__(self, x, sarimax_kwargs:dict, es_kwargs:dict):
        """
        Initialize the Model with time series data and model parameters.

        This constructor sets up SARIMAX and Exponential Smoothing models
        based on the provided time series data and parameters.

        Parameters:
        x (array-like): The time series data to be modeled.
        sarimax_kwargs (dict): Keyword arguments for SARIMAX model initialization.
        es_kwargs (dict): Keyword arguments for Exponential Smoothing model initialization.

        Returns:
        None

        Note:
        If model initialization fails due to TypeError, it's silently ignored.
        """
        self.x = x
        try:
            self.model_sarimax = sm.tsa.SARIMAX(x, **sarimax_kwargs).fit()
        except TypeError:
            pass

        try:
            self.model_es = ExponentialSmoothing(x, **es_kwargs).fit()
        except TypeError:
            pass

    def predict_arima(self, forecast_horizon:int)->np.array:
        """
        Predict future values using the SARIMAX model.

        This method generates predictions for the time series data using the SARIMAX model.
        It attempts to forecast values from the beginning of the time series up to the
        specified forecast horizon beyond the end of the original data.

        Parameters:
        forecast_horizon (int): The number of future time steps to forecast beyond
                                the end of the original time series.

        Returns:
        np.array: An array of predicted values, including both in-sample predictions
                and out-of-sample forecasts. If the SARIMAX model is not available
                (due to initialization failure), the method returns None.

        Note:
        If an AttributeError occurs (likely due to the SARIMAX model not being initialized),
        the method silently passes and returns None.
        """
        try:
            forecast_arima = self.model_sarimax.predict(start=0, end=len(self.x) + forecast_horizon)
            return forecast_arima
        except AttributeError:
            pass

    def predict_es(self, forecast_horizon:int)->np.array:
        """
        Predict future values using the Exponential Smoothing model.

        This method generates predictions for the time series data using the Exponential Smoothing model.
        It attempts to forecast values from the beginning of the time series up to the
        specified forecast horizon beyond the end of the original data.

        Parameters:
        forecast_horizon (int): The number of future time steps to forecast beyond
                                the end of the original time series.

        Returns:
        np.array: An array of predicted values, including both in-sample predictions
                and out-of-sample forecasts. If the Exponential Smoothing model is not available
                (due to initialization failure), the method returns None.

        Note:
        If an AttributeError occurs (likely due to the Exponential Smoothing model not being initialized),
        the method silently passes and returns None.
        """

        try:
            forecast_es = self.model_es.predict(start=0, end=len(self.x) + forecast_horizon)
            return forecast_es
        except AttributeError:
            pass
    
    def predict(self, forecast_horizon:int)->np.array:
        """
        Generate predictions using both SARIMAX and Exponential Smoothing models.

        This method combines predictions from both the SARIMAX and Exponential Smoothing
        models, forecasting values from the beginning of the time series up to the
        specified forecast horizon beyond the end of the original data.

        Parameters:
        forecast_horizon (int): The number of future time steps to forecast beyond
                                the end of the original time series.

        Returns:
        tuple of np.array: A tuple containing two numpy arrays:
                        - The first array contains predictions from the SARIMAX model.
                        - The second array contains predictions from the Exponential Smoothing model.
                        Each array includes both in-sample predictions and out-of-sample forecasts.

        Note:
        This method assumes that both SARIMAX and Exponential Smoothing models have been
        successfully initialized. If either model is not available, this method may raise
        an AttributeError.
        """
        try:
            forecast_arima = self.model_sarimax.predict(start=0, end=len(self.x) + forecast_horizon-1)
        except AttributeError:
            forecast_arima = self.model_es.predict(start=0, end=len(self.x) + forecast_horizon-1)
            
        try:
            forecast_es = self.model_es.predict(start=0, end=len(self.x) + forecast_horizon-1)
        except AttributeError:
            forecast_es = self.model_sarimax.predict(start=0, end=len(self.x) + forecast_horizon-1)
            
        return forecast_arima, forecast_es
        