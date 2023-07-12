import gc
from typing import Callable, List

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from market import MarketPair, with_mkt_price, TradeOrder
from utils import timer_func
from swap import price_impact_range


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
    strategy: Callable[[dict, MarketPair, dict], List[dict]],
    args: dict
) -> dict:
    gc.disable()
    trade_exec_info = []
    trades = trade_df.reset_index().to_dict(orient="records")
    for row in trades:
        mkt = with_mkt_price(mkt, row["price"])
        trade_exec_info.extend(strategy(row, mkt, args))
    gc.enable()
    return sim_results(trade_exec_info)


@timer_func
def peg_simulation(
    mkt: MarketPair,
    function: List[float],
    strategy: Callable[[TradeOrder, MarketPair, dict, datetime, bool], List[dict]],
    args: dict
) -> dict:
    gc.disable()
    trade_exec_info = []
    for idx in range(0, len(function) - 2):
        rel_gain = (function[idx + 1] - function[idx]) / function[idx]
        mkt = with_mkt_price(mkt, mkt.mkt_price * (1 + rel_gain))
        dx, dy = mkt.get_delta_reserves()
        if dy == 0 or dx == 0:
            continue
        trade = {}
        if dx > 0:
            trade = TradeOrder(mkt.ticker, dx, mkt.swap_fee)
        else:
            trade = TradeOrder(mkt.ticker, dy, mkt.swap_fee)

        trades = strategy(trade, mkt, args,
                          args['start_date'] + timedelta(seconds=args['timeframe']*idx), False)
        trade_exec_info.extend(trades)

        # do buybacks if possible
        sum_chain_profit = 0
        for trade in trades:
            sum_chain_profit += trade['chain_profit']
            trade['swap_pool'] = 0
            trade['staking_pool'] = 0
            trade['oracle_pool'] = 0
            trade['community_pool'] = 0            
            trade['buy_back_vol'] = 0
        if sum_chain_profit > 0:
            '''buy_back = TradeOrder(mkt.ticker, sum_chain_profit /
                                  mkt.mid_price, mkt.swap_fee)
            buy_back_trades = strategy(buy_back, mkt, args,
                                       args['start_date'] + timedelta(seconds=args['timeframe']*idx), True)
            for b in buy_back_trades:
                b['swap_pool'] = b['buy_back_vol'] * args['swap_pool_coef']
                b['staking_pool'] = b['buy_back_vol'] * args['staking_pool_coef']
                b['oracle_pool'] = b['buy_back_vol'] * args['oracle_pool_coef']
                b['community_pool'] = b['buy_back_vol'] * args['community_pool_coef']

            trade_exec_info.extend(buy_back_trades)'''
            a = 1

    gc.enable()
    return sim_results(trade_exec_info)


@ timer_func
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
    if 'cex_profit' in df_sim:
        df_sim["total_cex_profit"] = df_sim["cex_profit"].cumsum()
    if 'chain_profit' in df_sim:
        df_sim["total_chain_profit"] = df_sim["chain_profit"].cumsum()
    if 'buy_back_vol' in df_sim:
        df_sim["total_buy_back_vol"] = df_sim["buy_back_vol"].cumsum()
    if 'swap_pool' in df_sim:
        df_sim["total_swap_pool"] = df_sim["swap_pool"].cumsum()
    if 'staking_pool' in df_sim:
        df_sim["total_staking_pool"] = df_sim["staking_pool"].cumsum()
    if 'oracle_pool' in df_sim:
        df_sim["total_oracle_pool"] = df_sim["oracle_pool"].cumsum()
    if 'community_pool' in df_sim:
        df_sim["total_community_pool"] = df_sim["community_pool"].cumsum()

    return {
        "headline": trade_summary(df_sim),
        "breakdown": df_sim,
    }


@ timer_func
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
