import gc
from datetime import timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from .market import TradeOrder
from .strategy import DivStrategy, Strategy
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
    trade_df: pd.DataFrame,
    strategy: Strategy,
) -> dict:
    gc.disable()
    trade_exec_info = []
    trades = trade_df.reset_index().to_dict(orient="records")
    for row in trades:
        trade_exec_info.extend(strategy.execute(row, trade_df))
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
    df_sim.fillna(0, inplace=True)
    trade_data = df_sim[
        ["total_volume_base", "total_volume_quote", "total_fees_paid_quote"]
    ]
    df_sim[
        ["volume_base", "volume_quote", "fees_paid_quote"]
    ] = trade_data.diff().fillna(trade_data)
    df_sim["trade_pnl_pct"] = df_sim["trade_pnl"] / df_sim["hold_portfolio"]
    df_sim["fees_pnl_pct"] = df_sim["total_fees_paid_quote"] / df_sim["hold_portfolio"]
    df_sim["total_arb_profit"] = df_sim["arb_profit"].cumsum()

    if "cex_profit" in df_sim:
        df_sim["total_cex_profit"] = df_sim["cex_profit"].cumsum()
    if "chain_profit" in df_sim:
        df_sim["total_chain_profit"] = df_sim["chain_profit"].cumsum()
    if "buy_back_vol" in df_sim:
        df_sim["total_buy_back_vol"] = df_sim["buy_back_vol"].cumsum()
    if "swap_pool" in df_sim:
        df_sim["total_swap_pool"] = df_sim["swap_pool"].cumsum()
    if "staking_pool" in df_sim:
        df_sim["total_staking_pool"] = df_sim["staking_pool"].cumsum()
    if "oracle_pool" in df_sim:
        df_sim["total_oracle_pool"] = df_sim["oracle_pool"].cumsum()
    if "community_pool" in df_sim:
        df_sim["total_community_pool"] = df_sim["community_pool"].cumsum()
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


@timer_func
def peg_simulation(
    sim_function: dict,
    strategy: DivStrategy,
    args: dict,
    softpeg_strategy: str | None,
) -> dict:
    """_summary_

    Args:
        sim_function (dict): price function to simulate, array of floats
        strategy (DivStrategy): simulation strategy to be used
        args (dict): settings for the divergence strategy
        softpeg_strategy (str | None): 'time' or 'avg' accepted for softpeg raise

    Returns:
        dict: list of trades of the simulation with all info

    """
    gc.disable()
    trade_exec_info = []  # type: List[Dict]

    # store the starting soft peg price, as we possibly change it via softpeg_strategy
    args["soft_peg_start"] = args["soft_peg_price"]
    function = sim_function["data"]
    timeframe = sim_function["timeframe"]
    for idx in range(0, len(function) - 2):
        # check if we raise the softpeg
        if softpeg_strategy == "avg":
            strategy.avg_price_peg_increase(function, idx, trade_exec_info)
        elif softpeg_strategy == "time":
            strategy.time_peg_increase(function, idx)

        rel_gain = (function[idx + 1] - function[idx]) / function[idx]
        # ignore bigger than 50% drops, as noone would trade this cause tax is 100%
        if (rel_gain > 1 and rel_gain - 1 >= 0.5) or rel_gain >= 0.5:
            continue
        strategy.cp_amm.update_mkt_price(strategy.cp_amm.mkt.mkt_price * (1 + rel_gain))
        dx, dy = strategy.cp_amm.mkt.get_delta_reserves()
        if dy == 0 or dx == 0:
            continue
        if dx > 0:
            # dx = volume in USDT
            trade = TradeOrder(
                strategy.cp_amm.mkt.ticker, dx, strategy.cp_amm.mkt.swap_fee
            )
        else:
            # dy = volume in USTC
            trade = TradeOrder(
                strategy.cp_amm.mkt.ticker, dy, strategy.cp_amm.mkt.swap_fee
            )

        trades = strategy.execute_div(
            strategy.cp_amm.mkt.mkt_price * (1 + rel_gain),
            trade,
            args["start_date"] + timedelta(seconds=timeframe * idx),
            False,
        )
        trade_exec_info.extend(trades)

        # get sum of profits from tax in USDT
        sum_chain_profit = 0
        for t in trades:
            sum_chain_profit += t["chain_profit"]
            t["swap_pool"] = 0
            t["staking_pool"] = 0
            t["oracle_pool"] = 0
            t["community_pool"] = 0
            t["buy_back_vol"] = 0

        # if we have profits, do buyback and distribute USTC to pools
        if sum_chain_profit > 0 and args["do_buybacks"]:
            buy_back = TradeOrder(
                strategy.cp_amm.mkt.ticker,
                sum_chain_profit,
                strategy.cp_amm.mkt.swap_fee,
            )
            buy_back_trades = strategy.execute_div(
                strategy.cp_amm.mkt.mkt_price,
                buy_back,
                args["start_date"] + timedelta(seconds=timeframe * idx),
                True,
            )
            for b in buy_back_trades:
                b["swap_pool"] = b["buy_back_vol"] * args["swap_pool_coef"]
                b["staking_pool"] = b["buy_back_vol"] * args["staking_pool_coef"]
                b["oracle_pool"] = b["buy_back_vol"] * args["oracle_pool_coef"]
                b["community_pool"] = b["buy_back_vol"] * args["community_pool_coef"]
                b["cex_profit"] = 0  # we do not tax buybacks yet, see git issue
                b["chain_profit"] = 0  # we do not tax buybacks yet, see git issue
            trade_exec_info.extend(buy_back_trades)
            # trade_exec_info.extend(strategy.execute(row, trade_df))
    gc.enable()
    return sim_results(trade_exec_info)
