from datetime import datetime
from pytz import timezone

import yfinance as yf
from pandas import DataFrame

import ta
import requests

class BaseHelper():
    def __init__(self):
        pass

    def get_wbtc_price_dexscreener(self) -> float:
        """Fetches Wrapped Bitcoin (WBTC) price from DEX Screener API via Uniswap v3 WBTC/USDC pair on Ethereum.

        Returns:
            float: WBTC price in USD.
        """
        # WBTC/USDC pool CA address on Ethereum
        url = "https://api.dexscreener.com/latest/dex/pairs/ethereum/0x9a772018fbd77fcd2d25657e5c547baff3fd7d16"
        response = requests.get(url).json()
        price = float(response['pairs']['priceUsd'])
        return price


    def get_btc_price_yfinance(self, period: str, interval: str) -> DataFrame:
        """Fetches Bitcoin (BTC) price from yfinance.

        Args:
            str (period): Time period for price data (always "1d" for daily predictions).
            str (interval): Time interval for price data ("5m", "30m", "1h", "4h", or "24h").

        Returns:
            DataFrame: Historic BTC prices.
        """
        btc = yf.Ticker("BTC-USD")
        btc_hist = btc.history(period=period, interval=interval)
        return btc_hist
    
    def process_data_yfinance(self, period: str, interval: str, base_model:bool = False, drop_na:bool = False) -> DataFrame:
        data = self.get_btc_price_yfinance(period, interval)
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
        data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
        data['Momentum'] = ta.momentum.ROCIndicator(data['Close']).roc()
        if(base_model):
            for i in range(1,7):
                data[f'NextClose{i}'] = data['Close'].shift(-1*i)

        if(drop_na):
            data.dropna(inplace=True)

        data.reset_index(inplace=True)

        return data

    def get_est_time_now():
        ny_timezone = timezone("America/New_York")
        current_time_ny = datetime.now(ny_timezone)

        return current_time_ny


