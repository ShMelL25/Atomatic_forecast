from .opt_of_params import Optimizer
from .models import Model
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import datetime

class Forecaster:
    
    def __init__(self, 
                 data:np.array, 
                 data_index:np.array, 
                 project_name:str,
                 project_id:str,
                 n_trials:int = 10
                 ):
        """
        Initialize a Forecaster object with given data, index, project details, and number of trials for parameter optimization.

        Parameters:
        -----------
        data : np.array
            The historical data for forecasting.
        data_index : np.array
            The corresponding index for the historical data.
        project_name : str
            The name of the project for which the forecast is being made.
        project_id : str
            The unique identifier for the project.
        n_trials : int, optional
            The number of trials for parameter optimization in the model. Default is 10.

        Returns:
        --------
        None
        """
        self.data = data
        self.data_index = data_index
        self.project_id = project_id
        self.project_name = project_name
        self.kwargs_sarima, self.kwargs_es = Optimizer(data).opt_run(n_trials=n_trials)
        
        self.model = Model(x=self.data, sarimax_kwargs=self.kwargs_sarima, es_kwargs=self.kwargs_es)
        
    def predict(self, forecast_horizon: int) -> pd.DataFrame:
        """
        Generate forecasts for the specified horizon and return a formatted DataFrame.

        This method uses the ARIMA and Exponential Smoothing models to make predictions,
        averages their results, and formats the output into a structured DataFrame.

        Parameters:
        -----------
        forecast_horizon : int
            The number of future time periods to forecast.
            
        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the forecast results with the following columns:
            - 'Y': Year of the forecast
            - 'M': Month of the forecast
            - 'Project_Name': Name of the project
            - 'value': Forecasted value (rounded to integer)
            - 'Forecast_date': Date when the forecast was generated
            - 'Project_ID': Unique identifier of the project
        """
        pred_arima, pred_es = self.model.predict(forecast_horizon)
        
                
        df = pd.DataFrame(data={'date':self.generate_index(forecast_horizon), 
                                'value':(pred_arima+pred_es)/2,
                                'arima':pred_arima,
                                'es':pred_es}
                          ).iloc[-forecast_horizon:].set_index('date').reset_index()


        df['Project_Name'] = np.full(df.shape[0], self.project_name)
        df['Project_ID'] = np.full(df.shape[0], self.project_id)
        print(self.data_index.astype(str)[-1].replace('-',''))
        df['Forecast_date'] = np.full(df.shape[0], self.data_index.astype(str)[-1].replace('-',''))
        df['Y'], df['M'] = df['date'].astype(str).str.split('-', expand=True)[0], df['date'].astype(str).str.split('-', expand=True)[1]
        try:
            df['value'] = round(df['value'])
        except pd.errors.IntCastingNaNError:
            df['value'] = pred_es
        

        return df[['Y','M','Project_Name','value','Forecast_date','Project_ID', 'arima', 'es']]
     

    def generate_index(self, forecast_horizon: int) -> np.array:
        """
        Generate a new index array for forecasting.
        
        This method creates a new index array by extending the existing data index
        with future dates based on the specified forecast horizon.
        
        Parameters:
        -----------
        forecast_horizon : int
            The number of future time periods to generate indices for.
        
        Returns:
        --------
        np.array
            An array containing the original data index concatenated with
            the newly generated future dates.
        """
        forecast_index_arma  = [pd.to_datetime(self.data_index[-1])+ relativedelta(months=x) for x in range(1, forecast_horizon+1)]
        
        return pd.concat([pd.DataFrame(self.data_index.to_numpy()),pd.DataFrame(forecast_index_arma)]).to_numpy().T[0]
        