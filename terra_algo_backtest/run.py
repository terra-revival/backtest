from market import MarketQuote, new_market
from simulation import swap_simulation, peg_simulation
from plotting import new_div_simulation_figure
from strategy import get_strategy
import pandas as pd
from IPython.display import display
from IPython.core.display import HTML
from bokeh.io import output_notebook, show
import os
from brown import simulationSamples
from datetime import datetime
from strategy import div_protocol
base_pair, quote_pair = 'USTC/BUSD', 'USDT/BUSD'

# get trades from csv
df_ustc = pd.read_csv("E:/terra/terra_algo_backtest/terra_algo_backtest/ustc_busd_trades.csv", parse_dates=[
                      "trade_date"], index_col="trade_date")

# liquidity of BUSD
liquidity_usd = 10000000
ust_start_price = 0.01
base = MarketQuote(base_pair, df_ustc.price.iloc[0])
quote = MarketQuote(quote_pair, 1)
# end quote is USTC/USDT
binance_swap_fee = 0.12 / 100
# create a 1,000,000 USD market for LUNC/USTC with 1% swap fee
mkt = new_market(liquidity_usd, quote, base, binance_swap_fee)
# load simulations
sims = simulationSamples(42, False)
# run simulation
div_protocol_args = {
    "soft_peg_price": 0.02,
    "arbitrage_coef": 0.2,
    "cex_tax_coef": 0.4,
    "buy_backs_coef": 0.1,
    "timeframe": 3600,
    "start_date": datetime.now(),
    "swap_pool_coef": 0.475,
    "staking_pool_coef": 0.475,
    "oracle_pool_coef": 0.025,
    "community_pool_coef": 0.025
}
# check div protocol sanity
assert div_protocol_args['soft_peg_price'] > 0, "soft_peg should be > 0"
assert div_protocol_args['cex_tax_coef'] > 0 or div_protocol_args['cex_tax_coef'] < 1, "cex_tax_coef should be > 0 and < 1"
assert div_protocol_args['buy_backs_coef'] > 0 or div_protocol_args['buy_backs_coef'] < 1, "buy_backs_coef should be > 0 and < 1"
assert div_protocol_args['arbitrage_coef'] > 0 or div_protocol_args['arbitrage_coef'] < 1, "arbitrage_coef should be > 0 and < 1"

# run all prescripted simulations
for sim in sims:
    simul = peg_simulation(mkt, sims[sim], div_protocol, div_protocol_args)
    # display results
    show(new_div_simulation_figure(mkt, simul,
         plot_height=300, label_add=f'({sim} function)'))
