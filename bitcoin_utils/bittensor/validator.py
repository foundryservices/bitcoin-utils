from pandas import DataFrame


from base import BaseHelper

class Validator(BaseHelper):
    def __init__(self):
        pass

    def get_btc_price_bittensor_validator(self, interval:str, period: str) -> DataFrame:
        data = self.process_data_yfinance(interval, period)
        return data[-7:-1]