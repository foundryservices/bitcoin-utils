import os

from huggingface_hub import HfApi
from numpy import ndarray
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
from typing import Tuple

from base import BaseHelper

class Miner(BaseHelper):
    def __init__(self, hf_access_token:str = ""):
        if(hf_access_token != ""):
            if not os.getenv("HF_ACCESS_TOKEN"):
                print("Cannot find a Huggingface Access Token - unable to upload model to Huggingface.")
            else:
                self.hf_token = os.getenv("HF_ACCESS_TOKEN")
        

    def get_btc_price_bittensor_miner(self, interval:str, period: str) -> DataFrame:
        data = self.process_data_yfinance(interval, period)
        return data
    
    def scale_data(self, data:DataFrame) -> Tuple[MinMaxScaler, ndarray, ndarray]:
        """
        Normalize the data procured from yahoo finance between 0 & 1

        Function takes a dataframe as an input, scales the input and output features and
        then returns the scaler itself, along with the scaled inputs and outputs. Scaler is
        returned to ensure that the output being predicted can be rescaled back to a proper
        value.

        Input:
            :param data: The S&P 500 data procured from a certain source at a 5m granularity
            :type data: pd.DataFrame

        Output:
            :returns: A tuple of 3 values -
                    - scaler : which is the scaler used to scale the data (MixMaxScaler)
                    - X_scaled : input/features to the model scaled (np.ndarray)
                    - y_scaled : target variable of the model scaled (np.ndarray)
        """
        X = data[['Open', 'High', 'Low', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'CCI', 'Momentum']].values

        # Prepare target variable
        y = data[['NextClose1', 'NextClose2', 'NextClose3', 'NextClose4', 'NextClose5', 'NextClose6']].values

        y = y.reshape(-1, 6)

        # Scale features
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X)
        y_scaled = scaler.fit_transform(y)

        return scaler, X_scaled, y_scaled
    
    def upload_model_to_huggingface(self, model_name: str):
        api = HfApi()
        api.upload_file(
            path_or_fileobj="mining_models/base_lstm_new.h5",
            path_in_repo=f"{model_name}.h5",
            repo_id="foundryservices/bittensor-sn28-base-lstm",
            repo_type="model",
            token=self.hf_token
        )
    
    def create_and_save_base_model_lstm(self, scaler:MinMaxScaler, X_scaled:ndarray, y_scaled:ndarray) -> float:
        """
        Base model that can be created for predicting the S&P 500 close price

        The function creates a base model, given a scaler, inputs and outputs, and
        stores the model weights as a .h5 file in the mining_models/ folder. The model
        architecture and model name given now is a placeholder, can (and should)
        be changed by miners to build more robust models.

        Input:
            :param scaler: The scaler used to scale the inputs during model training process
            :type scaler: sklearn.preprocessing.MinMaxScaler

            :param X_scaled: The already scaled input data that will be used by the model to train and test
            :type X_scaled: np.ndarray

            :param y_scaled: The already scaled output data that will be used by the model to train and test
            :type y_scaled: np.ndarray
        
        Output:
            :returns: The MSE of the model on the test data
            :rtype: float
        """
        model_name = "mining_models/base_lstm_new"

        # Reshape input for LSTM
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

        # Split data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

        # LSTM model - all hyperparameters are baseline params - should be changed according to your required
        # architecture. LSTMs are also not the only way to do this, can be done using any algo deemed fit by
        # the creators of the miner.
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_scaled.shape[1], X_scaled.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=6))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=100, batch_size=32)
        model.save(f'{model_name}.h5')

        self.upload_model_to_huggingface()

        # Predict the prices - this is just for a local test, this prediction just allows
        # miners to assess the performance of their models on real data.
        predicted_prices = model.predict(X_test)

        # Rescale back to original range
        predicted_prices = scaler.inverse_transform(predicted_prices)
        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 6))

        # Evaluate
        mse = mean_squared_error(y_test_rescaled, predicted_prices)
        print(f'Mean Squared Error: {mse}')
        
        return mse

def create_and_save_base_model_regression(scaler:MinMaxScaler, X_scaled:ndarray, y_scaled:ndarray) -> float:
    """
    Base model that can be created for predicting the S&P 500 close price

    The function creates a base model, given a scaler, inputs and outputs, and
    stores the model weights as a .h5 file in the mining_models/ folder. The model
    architecture and model name given now is a placeholder, can (and should)
    be changed by miners to build more robust models.

    Input:
        :param scaler: The scaler used to scale the inputs during model training process
        :type scaler: sklearn.preprocessing.MinMaxScaler

        :param X_scaled: The already scaled input data that will be used by the model to train and test
        :type X_scaled: np.ndarray

        :param y_scaled: The already scaled output data that will be used by the model to train and test
        :type y_scaled: np.ndarray
    
    Output:
        :returns: The MSE of the model on the test data
        :rtype: float
    """
    model_name = "mining_models/base_linear_regression"

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # LSTM model - all hyperparameters are baseline params - should be changed according to your required
    # architecture. LSTMs are also not the only way to do this, can be done using any algo deemed fit by
    # the creators of the miner.
    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, f"{model_name}.joblib")

    # Predict the prices - this is just for a local test, this prediction just allows
    # miners to assess the performance of their models on real data.
    predicted_prices = model.predict(X_test)

    # Rescale back to original range
    predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Evaluate
    mse = mean_squared_error(y_test_rescaled, predicted_prices)
    print(f'Mean Squared Error: {mse}')
    
    return mse
    
    
    
