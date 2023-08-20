import gc

import numpy as np
import pandas as pd

from .strategy import Strategy
from .utils import timer_func


@timer_func
def swap_simulation(
    trade_df: pd.DataFrame,
    strategy: Strategy,
) -> dict:
    gc.disable()
    trade_exec_info = []
    trades = trade_df.reset_index().to_dict(orient="records")
    for row in trades:
        trade_exec_info.extend(strategy.execute(row, trade_df))
    gc.enable()
    return sim_results(strategy, trade_exec_info)


@timer_func
def sim_results(strategy: Strategy, sim_outputs: list) -> dict:
    """Processes simulation outputs to provide a structured result.

    Args:
        sim_outputs (list): The list of simulation outputs.

    Returns:
        dict: A dictionary containing the processed simulation results.

    """
    if len(sim_outputs) == 0:
        return {}
    df_sim = pd.DataFrame(sim_outputs).set_index("trade_date")
    df_sim.fillna(0, inplace=True)

    trade_data = df_sim[["total_fees_paid_quote"]]
    df_sim[["fees_paid_quote"]] = trade_data.diff().fillna(trade_data)
    df_sim["trade_pnl_pct"] = df_sim["trade_pnl"] / df_sim["hold_portfolio"]
    df_sim["fees_pnl_pct"] = df_sim["total_fees_paid_quote"] / df_sim["hold_portfolio"]
    df_sim["retail_volume_quote"] = df_sim["volume_quote"]
    if "arb_profit" in df_sim.columns:
        df_sim["retail_volume_quote"] -= df_sim["arb_volume_quote"]

    return {
        "headline": trade_summary(df_sim),
        "breakdown": resample_df(strategy, df_sim),
        "breakdown_trades": df_sim,
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


def resample_df(strategy, df_sim: pd.DataFrame) -> pd.DataFrame:
    """Resamples the DataFrame based on a given frequency.

    Args:
        df (pd.DataFrame): The DataFrame to be resampled.
        resample_freq (str): The resampling frequency.

    Returns:
        pd.DataFrame: The resampled DataFrame.

    """
    df = df_sim.loc[:, df_sim.columns != "side"]
    return df.resample('D').agg(strategy.agg(df_sim))

