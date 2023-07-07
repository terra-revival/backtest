=====
Usage
=====

Load trades from Binance::

    from binance import Client
    from terra_algo_backtest.binance_loader import new_binance_client

    # replace these with your Binance API key and secret
    client = new_binance_client(
        os.getenv("BINANCE_API_KEY"),
        os.getenv("BINANCE_API_SECRET"))
    # parameters
    params = {
        "base_pair": "LUNC/BUSD",
        "quote_pair": "USTC/BUSD",
        "start": "2023-03-01 00:00:00",
        "end": "2023-06-28 23:59:59",
        "frequency" = Client.KLINE_INTERVAL_1HOUR,
    }
    # create trades
    df_trades = client.create_trade_data(
        params["base_pair"],
        params["quote_pair"],
        params["vol_multiplier"],
        params["frequency"],
        params["start"],
        params["end"])

Initialise a new market to replay trades::

    from terra_algo_backtest.market import MarketQuote, new_market

    # market liquidity
    liquidity_usd = 100000
    # LUNC/BUSD market price
    base = MarketQuote(params["base_pair"], 0.00008)
    # USTC/BUSD market price
    quote = MarketQuote(params["quote_pair"], 0.01)
    # LUNC/USTC market with 100,000$ liquity and 0.3% tx fee
    mkt = new_market(liquidity_usd, quote, base, 0.003)

Replay trades with your algo::

    from terra_algo_backtest.strategy import get_strategy
    from terra_algo_backtest.simulation import swap_simulation

    # simple DEX liquidity provider strategy
    strategy = get_strategy("uni_v2")
    # replay trades against DEX
    simul = swap_simulation(mkt, df_trades, strategy)

Visualise simulation results::

    from terra_algo_backtest.plotting import new_simulation_figure

    # display results
    show(new_simulation_figure(mkt, simul, plot_height=300))

Tutorials
---------

    * How Uniswap works
    * How Terra Market Module works
    * LUNC/USTC back testing simulation

See docs/examples folder for interactive version using notebooks
