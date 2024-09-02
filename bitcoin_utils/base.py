from dotenv import load_dotenv
from pandas import DataFrame
import os
import requests

# import ta
# import yfinance as yf


class BaseHelper:
    def __init__(self) -> None:
        load_dotenv()
        self.CM_API_KEY = os.getenv("CM_API_KEY")

    def get_wbtc_price_dexscreener(self):
        """Fetches Wrapped Bitcoin (WBTC) price from DEX Screener API via Uniswap v3 WBTC/USDC pair on Ethereum.

        Returns:
            float: WBTC price in USD.
        """
        # WBTC/USDC pool CA address on Ethereum
        url = "https://api.dexscreener.com/latest/dex/pairs/ethereum/0x9a772018fbd77fcd2d25657e5c547baff3fd7d16"
        response = requests.get(url).json()
        print(response)
        return response

    def get_available_market_trades(self):
        """Fetches available market trades from CoinMetrics API.

        Returns:
            DataFrame: Available market trades
        """
        response = requests.get(
            f"https://api.coinmetrics.io/v4/catalog/market-trades?pretty=true&api_key={self.CM_API_KEY}"
        ).json()
        print(response)
        return response

    def get_available_market_quotes(self):
        """Fetches available market quotes from CoinMetrics API.

        Returns:
            DataFrame: Available market quotes
        """
        response = requests.get(
            f"https://api.coinmetrics.io/v4/catalog/market-quotes?pretty=true&api_key={self.CM_API_KEY}"
        ).json()
        print(response)
        return response

    def get_available_market_open_interest(self):
        """Fetches available market open interest from CoinMetrics API.

        Returns:
            DataFrame: Available market open interest
        """
        response = requests.get(
            f"https://api.coinmetrics.io/v4/catalog/market-openinterest?pretty=true&api_key={self.CM_API_KEY}"
        ).json()
        print(response)
        return response

    def get_available_market_candles(self):
        """Fetches available market candles from CoinMetrics API.

        Returns:
            DataFrame: Available market candles
        """
        response = requests.get(
            f"https://api.coinmetrics.io/v4/catalog/market-candles?pretty=true&api_key={self.CM_API_KEY}"
        ).json()
        print(response)
        return response

    def get_available_market_liquidations(self):
        """Fetches available market liquidations from CoinMetrics API.

        Returns:
            DataFrame: Available market liquidations
        """
        response = requests.get(
            f"https://api.coinmetrics.io/v4/catalog/market-liquidations?pretty=true&api_key={self.CM_API_KEY}"
        ).json()
        print(response)
        return response

    def get_available_market_contract_prices(self):
        """Fetches available market contract prices from CoinMetrics API.

        Returns:
            DataFrame: Available market contract prices
        """
        response = requests.get(
            f"https://api.coinmetrics.io/v4/catalog/market-contract-prices?pretty=true&api_key={self.CM_API_KEY}"
        ).json()
        print(response)
        return response

    def get_available_market_funding_rates(self):
        """Fetches available market funding rates from CoinMetrics API.

        Returns:
            DataFrame: Available market funding rates
        """
        response = requests.get(
            f"https://api.coinmetrics.io/v4/catalog/market-funding-rates?pretty=true&api_key={self.CM_API_KEY}"
        ).json()
        print(response)
        return response

    def get_available_market_implied_volatility(self):
        """Fetches available market implied volatility from CoinMetrics API.

        Returns:
            DataFrame: Available market implied volatility
        """
        response = requests.get(
            f"https://api.coinmetrics.io/v4/catalog/market-implied-volatility?pretty=true&api_key={self.CM_API_KEY}"
        ).json()
        print(response)
        return response

    def get_available_market_orderbooks(self):
        """Fetches available market orderbooks from CoinMetrics API.

        Returns:
            DataFrame: Available market orderbooks
        """
        response = requests.get(
            f"https://api.coinmetrics.io/v4/catalog/market-orderbooks?pretty=true&api_key={self.CM_API_KEY}"
        ).json()
        print(response)
        return response

    def get_available_market_greeks(self):
        """Fetches available market greeks from CoinMetrics API.

        Returns:
            DataFrame: Available market greeks
        """
        response = requests.get(
            f"https://api.coinmetrics.io/v4/catalog/market-greeks?pretty=true&api_key={self.CM_API_KEY}"
        ).json()
        print(response)
        return response

    def get_available_market_metrics(self):
        """Fetches available market metrics from CoinMetrics API.

        Returns:
            DataFrame: Available market metrics
        """
        response = requests.get(
            f"https://api.coinmetrics.io/v4/catalog/market-metrics?pretty=true&api_key={self.CM_API_KEY}"
        ).json()
        print(response)
        return response

    def get_available_asset_metrics(self):
        """Fetches available asset metrics from CoinMetrics API.

        Returns:
            DataFrame: Available asset metrics
        """
        response = requests.get(
            f"https://api.coinmetrics.io/v4/catalog/asset-metrics?pretty=true&api_key={self.CM_API_KEY}"
        ).json()
        print(response)
        return response

    def get_available_exchange_asset_pairs(self):
        """Fetches available exchange asset pairs from CoinMetrics API.

        Returns:
            DataFrame: Available exchange asset pairs
        """
        response = requests.get(
            f"https://api.coinmetrics.io/v4/catalog/exchange-assets?pretty=true&api_key={self.CM_API_KEY}"
        ).json()
        print(response)
        return response

    def get_available_exchange_asset_metrics(self):
        """Fetches available exchange asset metrics from CoinMetrics API.

        Returns:
            DataFrame: Available exchange asset metrics
        """
        response = requests.get(
            f"https://api.coinmetrics.io/v4/catalog/exchange-asset-metrics?pretty=true&api_key={self.CM_API_KEY}"
        ).json()
        print(response)
        return response

    def get_available_asset_pairs(self):
        """Fetches available asset pairs from CoinMetrics API.

        Returns:
            DataFrame: Available asset pairs
        """
        response = requests.get(
            f"https://api.coinmetrics.io/v4/catalog/pairs?pretty=true&api_key={self.CM_API_KEY}"
        ).json()
        print(response)
        return response

    # def get_btc_price_yfinance(self, period: str, interval: str) -> DataFrame:
    #     """Fetches Bitcoin (BTC) price from yfinance.

    #     Args:
    #         str (period): Time period for price data (always "1d" for daily predictions).
    #         str (interval): Time interval for price data ("5m", "30m", "1h", "4h", or "24h").

    #     Returns:
    #         DataFrame: Historic BTC prices.
    #     """
    #     btc = yf.Ticker("BTC-USD")
    #     btc_hist = btc.history(period=period, interval=interval)
    #     return btc_hist

    # def process_data_yfinance(
    #     self,
    #     period: str,
    #     interval: str,
    #     base_model: bool = False,
    #     drop_na: bool = False,
    # ) -> DataFrame:
    #     data = self.get_btc_price_yfinance(period, interval)
    #     data["SMA_50"] = data["Close"].rolling(window=50).mean()
    #     data["SMA_200"] = data["Close"].rolling(window=200).mean()
    #     data["RSI"] = ta.momentum.RSIIndicator(data["Close"]).rsi()
    #     data["CCI"] = ta.trend.CCIIndicator(
    #         data["High"], data["Low"], data["Close"]
    #     ).cci()
    #     data["Momentum"] = ta.momentum.ROCIndicator(data["Close"]).roc()
    #     if base_model:
    #         for i in range(1, 7):
    #             data[f"NextClose{i}"] = data["Close"].shift(-1 * i)

    #     if drop_na:
    #         data.dropna(inplace=True)

    #     data.reset_index(inplace=True)

    #     return data
