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

    df_sim_agg = strategy.agg_results(df_sim)

    return {
        "headline": trade_summary(df_sim),
        "breakdown": resample_df(df_sim_agg),
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

def resample_df(df_sim: pd.DataFrame) -> pd.DataFrame:
    """Resamples the DataFrame based on a given frequency.

    Args:
        df (pd.DataFrame): The DataFrame to be resampled.
        resample_freq (str): The resampling frequency.

    Returns:
        pd.DataFrame: The resampled DataFrame.

    """
    df = df_sim.loc[:, df_sim.columns != "side"]
    return df.resample('D').agg({
        'price': 'mean',
        "div_exec_price": "mean",

        'mid_price': 'last',
        "no_div_mid_price": "last",

        'mkt_price': 'last',
        'avg_price': 'mean',

        'spread': 'last',
        'price_impact': 'mean',
        'mkt_price_ratio': 'last',

        'impermanent_loss': 'last',

        'current_base': 'last',
        'current_quote': 'last',
        'cp_invariant': 'last',
        'asset_base_pct': 'last',

        'volume_base': 'sum',
        'volume_quote': 'sum',
        "div_volume_quote": "sum",
        "buy_back_volume_quote": "sum",

        'total_volume_base': 'last',
        'total_volume_quote': 'last',

        'fees_paid_quote': 'sum',
        'total_fees_paid_quote': 'last',

        'trade_pnl': 'last',
        'total_pnl': 'last',
        'trade_pnl_pct': 'last',
        'fees_pnl_pct': 'last',

        'roi': 'last',
        'hold_portfolio': 'last',
        'current_portfolio': 'last',

        'arb_profit': 'sum',
        'total_arb_profit': 'last',

        "div_tax_pct": "mean",
        "div_tax_quote": "sum",
        "reserve_account": "last",
    })
