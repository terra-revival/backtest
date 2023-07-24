import copy
import os
from datetime import datetime

import pandas as pd
from bokeh.io import output_notebook, show

from terra_algo_backtest.brown import simulationSamples
from terra_algo_backtest.market import MarketQuote, new_market
from terra_algo_backtest.plotting import new_simulation_figure
from terra_algo_backtest.simulation import peg_simulation
from terra_algo_backtest.strategy import (
    avg_price_peg_increase,
    div_protocol,
    time_peg_increase,
)

base_pair, quote_pair = "USTC/BUSD", "USDT/BUSD"

# liquidity of BUSD
liquidity_usd = 1000000
ust_start_price = 0.1
base = MarketQuote(base_pair, ust_start_price)
quote = MarketQuote(quote_pair, 1)
# end quote is USTC/USDT

# create a 1,000,000 USD market for LUNC/USTC with 0.12% swap fee
binance_swap_fee = 0.12 / 100
mkt = new_market(liquidity_usd, quote, base, binance_swap_fee)

# get default simulations
# 365 * 6 * 2 is data for 2 years 4h candles
sims = simulationSamples(42, False, 365 * 6 * 2)

# add scenario with real trades from kraken, M1 timeframe
# get trades from csv
df_ustc = pd.read_csv(
    "terra_algo_backtest/datasets/luna_kraken.csv",
    parse_dates=["date"],
    index_col="date",
)
sims["kraken_ustc/usdt_m1"] = {"data": df_ustc["close"], "timeframe": 60}

# run simulation
div_protocol_args = {
    "soft_peg_price": ust_start_price,  # start right at the peg price, seems a smart decision
    # after we reach final peg 1$, this ratio of tax is left for arb traders to keep. 0 = we tax 100% of divergence
    "arbitrage_coef": 0.05,
    "cex_tax_coef": 0.5,  # ratio of how much tax CEX gets
    # not used yet, we are not clear if to buyback with all the tax or keep base token also. 1 = spend all base tokens
    "buy_backs_coef": 1,
    "timeframe": 3600
    * 4,  # time difference between 2 price points of inputs in seconds, makes up axis X on charts. tied to steps in simulationSamples
    "start_date": datetime.now(),  # start date of axis X
    "swap_pool_coef": 0.475,  # ratio how much of tax goes to the swap pool
    "staking_pool_coef": 0.475,  # ratio how much of tax goes to the staking pool
    "oracle_pool_coef": 0.025,  # ratio how much of tax goes to the oracle pool
    "community_pool_coef": 0.025,  # ratio how much of tax goes to the community pool
    # soft peg related increase variables
    "soft_peg_increase": 0.1,  # how much is softpeg raised in 1 step
    "soft_peg_moving_window": 10,  # moving average window length used for price
}
# check div protocol sanity
assert div_protocol_args["soft_peg_price"] > 0, "soft_peg should be > 0"
assert (
    div_protocol_args["cex_tax_coef"] > 0 or div_protocol_args["cex_tax_coef"] < 1
), "cex_tax_coef should be > 0 and < 1"
assert (
    div_protocol_args["buy_backs_coef"] > 0 or div_protocol_args["buy_backs_coef"] < 1
), "buy_backs_coef should be > 0 and < 1"
assert (
    div_protocol_args["arbitrage_coef"] > 0 or div_protocol_args["arbitrage_coef"] < 1
), "arbitrage_coef should be > 0 and < 1"

# run all prescripted simulations
for sim in sims:
    # we do a deep copy to start each simulation with original market state
    # use_market = copy.deepcopy(mkt)
    base = MarketQuote(base_pair, ust_start_price)
    quote = MarketQuote(quote_pair, 1)
    mkt = new_market(liquidity_usd, quote, base, binance_swap_fee)

    use_args = copy.deepcopy(div_protocol_args)
    simul = peg_simulation(
        mkt, sims[sim], div_protocol, use_args, time_peg_increase
    )
    simul["sim_name"] = sim
    # display results
    show(new_simulation_figure(mkt, simul))
