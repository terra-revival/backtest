===================
Terra Algo Backtest
===================


.. image:: https://img.shields.io/pypi/v/terra_algo_backtest.svg
        :target: https://pypi.python.org/pypi/terra_algo_backtest

.. image:: https://img.shields.io/travis/terra-revival/terra_algo_backtest.svg
        :target: https://travis-ci.com/terra-revival/terra_algo_backtest

.. image:: https://readthedocs.org/projects/terra_algo_backtest/badge/?version=latest
        :target: https://terra_algo_backtest.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/terra-revival/terra_algo_backtest/shield.svg
     :target: https://pyup.io/repos/github/terra-revival/terra_algo_backtest/
     :alt: Updates



A package to replay trades and simulate strategies

* Documentation: https://terra_algo_backtest.readthedocs.io
* GitHub: https://github.com/terra-revival/terra_algo_backtest
* Free and open source software: `Apache Software License 2.0 <https://github.com/terra-revival/terra_algo_backtest/blob/main/LICENSE>`_


Features
--------

.. |check| raw:: html

    <input checked=""  disabled="" type="checkbox">

.. |uncheck| raw:: html

    <input disabled="" type="checkbox">

- |check| Load trades from CEX
- |uncheck| Load trades from DEX
- |check| Support custom algo
- |uncheck| Replay trades against CEX
- |check| Replay trades against DEX
- |check| Display simulation metrics


.. figure:: terra_backtest_algo.png
   :scale: 60 %


Quick Start
-----------

**1 - Install from source**

.. code-block:: console

    $ git clone git://github.com/terra-revival/terra_algo_backtest
    $ cd ./terra_algo_backtest
    $ pip install -r requirements_dev.txt
    $ make dist install
    $ cd ./docs/examples
    $ jupyter notebook

**2 - Load trades from Binance**

.. code-block:: python

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

**3 - Initialise a new market**

.. code-block:: python

    from terra_algo_backtest.market import MarketQuote, new_market

    # market liquidity
    liquidity_usd = 100000
    # LUNC/BUSD market price
    base = MarketQuote(params["base_pair"], 0.00008)
    # USTC/BUSD market price
    quote = MarketQuote(params["quote_pair"], 0.01)
    # LUNC/USTC market with 100,000$ liquity and 0.3% tx fee
    mkt = new_market(liquidity_usd, quote, base, 0.003)

**4 - Replay trades with your algo**

.. code-block:: python

    from terra_algo_backtest.strategy import get_strategy
    from terra_algo_backtest.simulation import swap_simulation

    # simple DEX liquidity provider strategy
    strategy = get_strategy("uni_v2")
    # replay trades against DEX
    simul = swap_simulation(mkt, df_trades, strategy)

**5 - Visualise simulation results**

.. code-block:: python

    from terra_algo_backtest.plotting import new_simulation_figure

    # display results
    show(new_simulation_figure(mkt, simul, plot_height=300))

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
