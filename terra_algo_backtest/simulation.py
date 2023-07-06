import gc
from typing import Callable, List

import numpy as np
import pandas as pd

from .market import MarketPair, with_mkt_price
from .utils import timer_func


def resample_df(df: pd.DataFrame, resample_freq: str) -> pd.DataFrame:
    """Resamples the DataFrame based on a given frequency.

    Args:
        df (pd.DataFrame): The DataFrame to be resampled.
        resample_freq (str): The resampling frequency.

    Returns:
        pd.DataFrame: The resampled DataFrame.

    """
    df = df.resample(resample_freq).agg(
        {
            "marketprice": np.mean,
            "buy_volume": np.sum,
            "sell_volume": np.sum,
            "fx_pnl": np.mean,
            "closeprice_x": np.mean,
            "closeprice_y": np.mean,
        }
    )
    df = df.dropna()
    return df


@timer_func
def swap_simulation(
    mkt: MarketPair,
    trade_df: pd.DataFrame,
    strategy: Callable[[dict, MarketPair], List[dict]],
) -> dict:
    gc.disable()
    trade_exec_info = []
    trades = trade_df.reset_index().to_dict(orient="records")
    for row in trades:
        mkt = with_mkt_price(mkt, row["price"])
        trade_exec_info.extend(strategy(row, mkt))
    gc.enable()
    return sim_results(trade_exec_info)


@timer_func
def sim_results(sim_outputs: list) -> dict:
    """Processes simulation outputs to provide a structured result.

    Args:
        sim_outputs (list): The list of simulation outputs.

    Returns:
        dict: A dictionary containing the processed simulation results.

    """
    if len(sim_outputs) == 0:
        return {}
    df_sim = pd.DataFrame(sim_outputs).set_index("trade_date")
    trade_data = df_sim[
        ["total_volume_base", "total_volume_quote", "total_fees_paid_quote"]
    ]
    df_sim[
        ["volume_base", "volume_quote", "fees_paid_quote"]
    ] = trade_data.diff().fillna(trade_data)
    df_sim["trade_pnl_pct"] = df_sim["trade_pnl"] / df_sim["hold_portfolio"]
    df_sim["fees_pnl_pct"] = df_sim["total_fees_paid_quote"] / df_sim["hold_portfolio"]
    df_sim["total_arb_profit"] = df_sim["arb_profit"].cumsum()

    return {
        "headline": trade_summary(df_sim),
        "breakdown": df_sim,
    }


@timer_func
def trade_summary(df_trades: pd.DataFrame) -> pd.DataFrame:
    """Generates a summary of trades.

    Args:
        df_trades (pd.DataFrame): The DataFrame containing trade data.

    Returns:
        pd.DataFrame: A DataFrame summarizing the trades.

    """
    df = df_trades[["side", "volume_base", "volume_quote"]].groupby("side")
    df = df.agg(["count", "sum"]).T.droplevel(1).drop_duplicates()
    df.index = ["Number of trades", "volume_base", "volume_quote"]
    df["total"] = df["buy"] + df["sell"]
    df.loc["avg_price"] = -df.loc["volume_quote"] / df.loc["volume_base"]
    df.columns.name = None
    return df
