# #!/usr/bin/env python

# """Tests for `terra_algo_backtest` package."""

# import numpy as np
# import pandas as pd
# import pandas.api.types as ptypes
# import pytest

# from terra_algo_backtest.market import (
#     MarketPair,
#     MarketQuote,
#     Pool,
#     TradeOrder,
#     new_market,
# )
# from terra_algo_backtest.simulation import (
#     swap_simulation,
# )
# from terra_algo_backtest.swap import (
#     constant_product_curve,
#     constant_product_swap,
# )
# from terra_algo_backtest.utils import with_debug_info


# def test_pool_creation():
#     """Tests creation of a pool."""
#     ticker = "A"
#     reserve = 100
#     pool = Pool(ticker, reserve)
#     assert pool.ticker == ticker
#     assert pool.balance == reserve
#     assert pool.reserves == [reserve]
#     assert pool.initial_deposit == reserve


# def test_mkt_creation():
#     """Tests creation of a market pair."""
#     pool_1 = Pool("A", 100)
#     pool_2 = Pool("B", 10)
#     mkt = MarketPair(pool_1, pool_2, 0)
#     assert mkt.ticker == "A/B"
#     assert mkt.pool_1.balance == 100
#     assert mkt.pool_2.balance == 10


# def test_mkt_get_reserves():
#     """Tests creation of a market pair."""
#     pool_1 = Pool("A", 100)
#     pool_2 = Pool("B", 10)
#     mkt = MarketPair(pool_1, pool_2, 0)
#     x, y = mkt.get_reserves(mkt.ticker)
#     assert x == 100
#     assert y == 10
#     x, y = mkt.get_reserves("B/A")
#     assert x == 10
#     assert y == 100


# @pytest.mark.parametrize(
#     "reserve_1,reserve_2",
#     [
#         (10, 2),
#         (100, 100),
#         (1000, 1000),
#         (10000, 10000),
#         (100000, 100000),
#         (1000000, 1000000),
#         (10000000, 10000000),
#         (100000000, 100000000),
#         (1000000000, 1000000000),
#         (134566.678899, 134566.67889927),
#         (0.333333333333333, 0.333333333333333),
#         (0.333333333333333, 0.1111111111111110),
#         (1000000000.033647474859, 1000000000.039484859),
#         (10000000000000.333333333333333, 10000000000000.67889927),
#     ],
# )
# def test_constant_product_curve(reserve_1, reserve_2):
#     """Tests that the constant product curve remains invariant in the XY curve produced
#     by constant_product_curve."""
#     x, y = constant_product_curve(
#         MarketPair(Pool("A", reserve_1), Pool("B", reserve_2), 0),
#         x_min=0.1 * reserve_1,
#         x_max=10.0 * reserve_2,
#         num=1000,
#     )
#     assert len(x) == len(y) == 1000
#     k_actual = np.multiply(x, y)
#     k_expected = [reserve_1 * reserve_2] * len(k_actual)
#     assert np.allclose(k_expected, k_actual, rtol=1e-14)


# @pytest.mark.parametrize(
#     "reserve_1,reserve_2",
#     [
#         (10, 2),
#         (100, 100),
#         (1000, 1000),
#         (10000, 10000),
#         (100000, 100000),
#         (1000000, 1000000),
#         (10000000, 10000000),
#         (100000000, 100000000),
#         (1000000000, 1000000000),
#         (134566.678899, 134566.67889927),
#         (0.333333333333333, 0.333333333333333),
#         (0.333333333333333, 0.1111111111111110),
#         (1000000000.033647474859, 1000000000.039484859),
#         (10000000000000.333333333333333, 10000000000000.67889927),
#     ],
# )
# def test_constant_product_swap(reserve_1, reserve_2):
#     """Tests that swaping produces the same curve as constant_product_curve."""
#     x, y = constant_product_curve(
#         MarketPair(Pool("A", reserve_1), Pool("B", reserve_2), 0),
#         x_min=0.01 * reserve_1,
#         x_max=10.0 * reserve_2,
#         num=10000,
#     )

#     i = 0
#     dx = np.diff(x)
#     x_actual, y_actual = [x[0]], [y[0]]
#     mkt = MarketPair(Pool("A", x[0]), Pool("B", y[0]), 0)
#     while i < len(dx):
#         order = TradeOrder(mkt.ticker, dx[i], 0)
#         dy, _ = constant_product_swap(mkt, order)
#         x_actual.append(x_actual[i] + dx[i])
#         y_actual.append(y_actual[i] - dy)
#         i = i + 1
#     assert np.allclose(x_actual, x, rtol=1e-14)
#     assert np.allclose(y_actual, y, rtol=1e-14)


# def test_constant_product_swap_fee():
#     """Tests that swaping produces the same curve as constant_product_curve."""
#     reserve_1, reserve_2 = 100, 100
#     mkt = MarketPair(Pool("A", reserve_1), Pool("B", reserve_2), 0.01)
#     order = TradeOrder(mkt.ticker, 100, mkt.swap_fee)
#     constant_product_swap(mkt, order)
#     assert mkt.pool_1.transaction_fees[0] == 1.0
#     assert mkt.pool_1.cumulated_transaction_fees[0] == 1.0
#     constant_product_swap(mkt, order)
#     assert mkt.pool_1.transaction_fees[1] == 1.0
#     assert mkt.pool_1.cumulated_transaction_fees[1] == 2.0


# def test_create_buy_sell_orders():
#     percentage_volume = 0.01
#     mkt = MarketPair(Pool("A", 100), Pool("B", 10), 0.01)
#     df = get_binance_trade_histo_for_pair(
#         "./tests/LUNCBUSD_1m.txt", "./tests/USTCBUSD_1m.txt"
#     )
#     orders = create_buy_sell_orders(mkt, df, percentage_volume, 0.5)
#     data = zip(
#         orders,
#         df["trading_pair_1_volume"].to_list(),
#         df["trading_pair_2_volume"].to_list(),
#     )
#     for order, volume_1, volume_2 in data:
#         volume = volume_1 if order.ticker == mkt.ticker else volume_2
#         assert percentage_volume * volume == order.order_size


# def test_simulation():
#     mkt = new_market(
#         10000000,
#         MarketQuote("LUNC/BUSD", 0.00024019),
#         MarketQuote("USTC/BUSD", 0.04153756),
#         0.003,
#     )
#     df = get_binance_trade_histo_for_pair(
#         "./docs/examples/LUNCBUSD_1m.txt", "./docs/examples/USTCBUSD_1m.txt"
#     )
#     df = df.loc["2022-11-01 00:10:00":"2022-11-02 00:10:59"]
#     sim_res_1, sim_res_2, df_debug = swap_simulation(mkt, df, 0.01, 0.5, True)
#     print(df_debug)

'''def test_simulationSamples():
    functions = simulationSamples(1420, False, 365, 24*60*60)
    assert type(functions) == dict
    for sim in functions:
        assert 'data' in functions[sim]
        assert 'timeframe' in functions[sim]
        assert functions[sim]['timeframe'] == 24*60*60
        assert len(functions[sim]['data']) == 365 + 1


def test_calc_taxes():
    tests = [
        {"args": {"soft_peg_price": 0.1, "arbitrage_coef": 0, "cex_tax_coef": 0},
         "price": 0.1, "volume": 1000, "output": [0, 0]},
        {"args": {"soft_peg_price": 0.2, "arbitrage_coef": 0, "cex_tax_coef": 0.5},
            "price": 0.1, "volume": 1000, "output": [500, 500]},
        {"args": {"soft_peg_price": 0.2, "arbitrage_coef": 0, "cex_tax_coef": 0.4},
            "price": 0.04, "volume": 1000, "output": [400, 600]},
        {"args": {"soft_peg_price": 0.2, "arbitrage_coef": 0, "cex_tax_coef": 0.5},
            "price": 0.3, "volume": 1000, "output": [0, 0]},
        {"args": {"soft_peg_price": 1, "arbitrage_coef": 0, "cex_tax_coef": 0.4},
            "price": 1.1, "volume": 1000, "output": [80, 120]},
        {"args": {"soft_peg_price": 1, "arbitrage_coef": 0.2, "cex_tax_coef": 0.5},
            "price": 0.8, "volume": 1000, "output": [160, 160]},
    ]
    for test in tests:
        cex, chain = calc_taxes(test["price"], test["volume"], test["args"])
        assert cex == test["output"][0]
        assert chain == test["output"][1]
'''
