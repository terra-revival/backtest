import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from binance.client import Client
from bokeh.io import push_notebook, show
from bokeh.layouts import column
from bokeh.models import Div
from joblib import Memory
from tqdm.auto import tqdm

from .market import split_ticker
from .utils import CacheDataFrame


def new_binance_client(api_key, secret):
    loading_div = Div(
        text="<div style='font-size: 14px; text-align: center;'>Loading...</div>"
    )
    handle = show(column(loading_div), notebook_handle=True)
    client = BinanceClient(api_key, secret)
    if client.status == "connected":
        loading_div.text = """
        <div style='font-size: 14px; text-align: center;'>
        <i class='fa fa-check-circle' style='color: green;'></i>
        &nbsp Connected to Binance
        </div>"""
    else:
        loading_div.text = """
        <div style='font-size: 14px; text-align: left;'>
        <i class='fa fa-exclamation-circle' style='color: red;'></i>
        &nbsp Binance client disconnected
        </div>"""
        loading_div.text += f"""<div style="
            width: 1000px;
            border: 1px solid red;
            color: red;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            background-color: #f8d7da;
            text-align: left;
            font-size: 14px;">
            {client.status}
        </div>"""

    push_notebook(handle=handle)
    return client


class BinanceClient:
    def __init__(self, api_key, api_secret):
        try:
            self.client = Client(api_key, api_secret)
            self.status = "connected"
        except Exception as e:
            self.status = f"{str(e)}"

        self.mem_cache = Memory("cache/")
        self.api_key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        self.api_secret_hash = hashlib.sha256(api_secret.encode("utf-8")).hexdigest()
        self.get_trade_data = self.mem_cache.cache(self.get_trade_data)
        self.create_trade_data = self.mem_cache.cache(self.create_trade_data)

    def __repr__(self):
        return (
            f"BinanceClient("
            f"api_key_hash='{self.api_key_hash}', "
            f"api_secret_hash='{self.api_secret_hash}')"
        )

    def __reduce__(self):
        return (self.__class__, (self.api_key_hash, self.api_secret_hash))

    # Function to fetch klines for a specific interval
    def fetch_klines(self, symbol, interval, start_time, end_time):
        klines = self.client.get_historical_klines(
            symbol, interval, start_time, end_time
        )
        return klines

    # Split the period into intervals and run queries per interval in parallel
    def fetch_klines_parallel(
        self, symbol, interval, start_time, end_time, n_intervals=10
    ):
        start_timestamp = pd.Timestamp(start_time)
        end_timestamp = pd.Timestamp(end_time) if end_time else pd.Timestamp.now()

        interval_duration = (end_timestamp - start_timestamp) / n_intervals

        intervals = [
            (
                start_timestamp + i * interval_duration,
                start_timestamp + (i + 1) * interval_duration,
            )
            for i in range(n_intervals)
        ]

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.fetch_klines, symbol, interval, str(start), str(end)
                )
                for start, end in intervals
            ]
            results = [
                future.result()
                for future in tqdm(
                    futures, desc="Fetching klines from Binance", total=len(futures)
                )
            ]

        klines = [item for sublist in results for item in sublist]
        return klines

    def get_trade_data(
        self, symbol: str, interval: str, start_time: str, end_time: str = None
    ) -> CacheDataFrame:
        base_asset, quote_asset = split_ticker(symbol)
        klines = self.fetch_klines_parallel(
            symbol.replace("/", ""), interval, start_time, end_time
        )

        # Convert klines to a DataFrame
        ohlcv_data = pd.DataFrame(
            klines,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_volume",
                "taker_buy_quote_volume",
                "ignore",
            ],
        )
        ohlcv_data["timestamp"] = pd.to_datetime(ohlcv_data["timestamp"], unit="ms")
        ohlcv_data[
            ["open", "high", "low", "close", "volume", "taker_buy_base_volume"]
        ] = ohlcv_data[
            ["open", "high", "low", "close", "volume", "taker_buy_base_volume"]
        ].apply(
            pd.to_numeric
        )
        # Calculate the market sell volume
        ohlcv_data["market_sell_volume"] = (
            ohlcv_data["volume"] - ohlcv_data["taker_buy_base_volume"]
        )
        # Calculate the net market volume
        ohlcv_data["net_market_volume"] = (
            ohlcv_data["taker_buy_base_volume"] - ohlcv_data["market_sell_volume"]
        )
        # Determine the trade direction
        ohlcv_data["direction"] = np.where(
            ohlcv_data["net_market_volume"] >= 0, "buy", "sell"
        )
        # Determine the asset unit
        ohlcv_data["asset_unit"] = np.where(
            ohlcv_data["direction"] == "buy", quote_asset, base_asset
        )
        # Calculate the quantity
        ohlcv_data["quantity"] = np.where(
            ohlcv_data["direction"] == "buy",
            ohlcv_data["net_market_volume"] * ohlcv_data["close"],
            ohlcv_data["net_market_volume"],
        )
        # Determine the direction price
        ohlcv_data["direction_price"] = np.where(
            ohlcv_data["close"] >= ohlcv_data["open"], "buy", "sell"
        )
        # Select relevant columns and set the index
        trade_data = ohlcv_data[
            [
                "timestamp",
                "close",
                "quantity",
                "direction",
                "direction_price",
                "asset_unit",
            ]
        ].rename(columns={"timestamp": "trade_date", "close": "price"})
        trade_data = trade_data.set_index("trade_date")
        trade_data = trade_data.astype(
            {
                "price": "float64",
                "quantity": "float64",
                "direction": "string",
                "direction_price": "string",
                "asset_unit": "string",
            }
        )
        return CacheDataFrame(trade_data)

    def create_trade_data(
        self,
        symbol1: str,
        symbol2: str,
        pct_volume: float,
        interval: str,
        start_time: str,
        end_time: str = None,
    ) -> CacheDataFrame:
        base_asset, pivot_1 = split_ticker(symbol1)
        quote_asset, pivot_2 = split_ticker(symbol2)

        # Ensure both symbols share the same quote asset
        if pivot_1 != pivot_2:
            raise ValueError("Symbols must share the same quote asset.")

        trade_data1 = self.get_trade_data(symbol1, interval, start_time, end_time)
        trade_data2 = self.get_trade_data(symbol2, interval, start_time, end_time)
        trade_data = trade_data1.join(
            trade_data2, how="inner", lsuffix="_1", rsuffix="_2"
        ).dropna()
        trades = (
            trade_data.query("direction_1 != direction_2")
            .reset_index()
            .to_dict(orient="records")
        )
        for row in tqdm(trades, total=len(trades), desc="Creating trades"):
            price1 = row["price_1"]
            price2 = row["price_2"]
            quantity1 = abs(row["quantity_1"])
            quantity2 = abs(row["quantity_2"])
            direction1 = row["direction_1"]
            direction2 = row["direction_2"]

            # Determine the composite direction
            if direction1 == "buy" and direction2 == "sell":
                row["quantity"] = (
                    pct_volume * min(quantity1, quantity2 * price2) / price2
                )
                row["asset_unit"] = quote_asset
            elif direction1 == "sell" and direction2 == "buy":
                row["quantity"] = (
                    -pct_volume * min(quantity1 * price2, quantity2) / price1
                )
                row["asset_unit"] = base_asset
            else:
                raise ValueError(
                    "Error with df query `direction_1 != direction_2`\nRow: {}".format(
                        row
                    )
                )

        composite_volume = pd.DataFrame(trades).set_index("trade_date")
        composite_volume["price"] = (
            composite_volume["price_1"] / composite_volume["price_2"]
        )
        composite_volume["direction"] = composite_volume["direction_1"]
        composite_volume = composite_volume.astype(
            {
                "price": "float64",
                "quantity": "float64",
                "direction": "string",
                "asset_unit": "string",
                "price_1": "float64",
                "quantity_1": "float64",
                "direction_1": "string",
                "direction_price_1": "string",
                "asset_unit_1": "string",
                "price_2": "float64",
                "quantity_2": "float64",
                "direction_2": "string",
                "direction_price_2": "string",
                "asset_unit_2": "string",
            }
        )

        return CacheDataFrame(composite_volume)
