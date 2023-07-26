import pytest
from pytest import approx

from terra_algo_backtest.exec_engine import (
    ConstantProductEngine,
    calc_arb_trade,
    calc_arb_trade_pnl,
)
from terra_algo_backtest.market import MarketPair, Pool, TradeOrder

# Conventions
# - long trade = buy pool / sell mkt eg. buy A/B pool / sell A/B mkt
# - short trade = sell pool / buy mkt eg. sell A/B pool / buy A/B mkt

long_trade = TradeOrder("A/B", 1000, 0)
short_trade = TradeOrder("B/A", -1000, 0)

expected_exec_info = {
    "long": {
        "trade_date": "2023-07-22",
        "side": "buy",
        "price": 3.0,
        "price_impact": -0.5,
        "qty_received": 333.3333333333333,
        "mid_price": 4.499999999999999,
        "mkt_price": 2,
        "spread": -2.499999999999999,
        "avg_price": 3.0303030303030307,
        "current_base": 666.6666666666667,
        "current_quote": 3000.0,
        "cp_invariant": 2000000.0000000002,
        "total_fees_paid_quote": 10.1010101010101,
        "total_volume_base": -333.3333333333333,
        "total_volume_quote": 1010.1010101010102,
        "asset_base_pct": 0.49999999999999994,
        "hold_portfolio": 6489.898989898989,
        "current_portfolio": 6000.0,
        "trade_pnl": -489.8989898989894,
        "total_pnl": -479.7979797979793,
        "roi": -0.0739299610894941,
        "impermanent_loss": 0.0,
        "mkt_price_ratio": 1.0,
    },
    "short": {
        "trade_date": "2023-07-22",
        "side": "sell",
        "price": 1.0,
        "price_impact": 0.5,
        "qty_received": 1000.0,
        "mid_price": 0.5,
        "mkt_price": 2,
        "spread": 1.5,
        "avg_price": 1.0101010101010102,
        "current_base": 2000.0,
        "current_quote": 1000.0,
        "cp_invariant": 2000000.0,
        "total_fees_paid_quote": 10.1010101010101,
        "total_volume_base": 1000.0,
        "total_volume_quote": -1010.1010101010102,
        "asset_base_pct": 0.5,
        "hold_portfolio": 2510.10101010101,
        "current_portfolio": 2000.0,
        "trade_pnl": -510.10101010101016,
        "total_pnl": -500.00000000000006,
        "roi": -0.19919517102615697,
        "impermanent_loss": 0.0,
        "mkt_price_ratio": 1.0,
    },
}


def assert_exec_info(trade_exec_info, expected_exec_info):
    for key in expected_exec_info.keys():
        assert key in trade_exec_info.keys()
        assert trade_exec_info[key] == expected_exec_info[key]


# def test_get_arb_implied_trade_order():
#     mkt = new_market(
#         10000000,
#         MarketQuote("LUNC/BUSD", 0.00024019),
#         MarketQuote("USTC/BUSD", 0.04153756),
#         0.01,
#     )
#     df = get_binance_trade_histo_for_pair(
#         "./docs/examples/LUNCBUSD_1m.txt", "./docs/examples/USTCBUSD_1m.txt"
#     )
#     for _, row in df.iterrows():
#         order_arb = get_arb_trade_order(mkt, row["price"])
#         if order_arb.order_size > 0:
#             assert abs((mkt.mid_price - row["price"]) / mkt.mid_price) > 0
#             constant_product_swap(mkt, order_arb)
#             assert (mkt.mid_price - row["price"]) / mkt.mid_price < 1e-2


@pytest.mark.parametrize(
    "mkt_price",
    [
        (30_000),
        (29_000),
        (31_000),
        (10_000),
        (60_000),
    ],
)
def test_calc_arb_trade(mkt_price):
    cp_amm = ConstantProductEngine(
        mkt=MarketPair(
            pool_1=Pool("USD", 3_000_000),
            pool_2=Pool("BTC", 100),
            swap_fee=0,
            mkt_price=mkt_price,
        )
    )

    ctx = {"trade_date": "2023-07-22", "price": mkt_price}

    arb_trade, exec_price = calc_arb_trade(cp_amm)
    if mkt_price != cp_amm.mkt.mid_price:
        assert arb_trade is not None
        exec_info = cp_amm.execute_trade(ctx, arb_trade)

        assert_exec_info(
            exec_info,
            {
                "trade_date": "2023-07-22",
                "side": "sell" if mkt_price < 30_000 else "buy",
                "total_fees_paid_quote": 0.0,
                "mkt_price": approx(mkt_price),
                "mid_price": approx(mkt_price),
            },
        )

    else:
        assert arb_trade is None


@pytest.mark.parametrize(
    "trade, pool_exec_price, mkt_price, fees, expected_pnl",
    [
        (long_trade, 10, 20, 0, 1000),  # long pool @ 10 / short mkt @ 20 / no fees
        (long_trade, 10, 20, 0.01, 990),  # long pool @ 10 / short mkt @ 20 / 1% fees
        (long_trade, 20, 10, 0, -500),  # long pool @ 20 / short mkt @ 10 / no fees
        (long_trade, 20, 10, 0.01, -510),  # long pool @ 20 / short mkt @ 10 / 1% fees
        (short_trade, 20, 10, 0, 1000),  # long mkt @ 10 / short pool @ 20 / no fees
        (short_trade, 20, 10, 0.01, 990),  # long mkt @ 10 / short pool @ 20 / 1% fees
        (short_trade, 10, 20, 0, -500),  # long mkt @ 20 / short pool @ 10 / no fees
        (short_trade, 10, 20, 0.01, -510),  # long mkt @ 20 / short pool @ 10 / 1% fees
    ],
)
def test_calc_arb_trade_pnl(trade, pool_exec_price, mkt_price, fees, expected_pnl):
    pnl = calc_arb_trade_pnl(trade, pool_exec_price, mkt_price, fees)
    assert pnl == expected_pnl


@pytest.mark.parametrize(
    "trade, pool_exec_price, mkt_price, fees",
    [
        (long_trade, 0, 20, 0.002),  # Test when pool_exec_price is zero
        (short_trade, 20, 0, 0.002),  # Test when mkt_price is zero
    ],
)
def test_calc_arb_trade_pnl_zero_price(trade, pool_exec_price, mkt_price, fees):
    with pytest.raises(ValueError):
        calc_arb_trade_pnl(trade, pool_exec_price, mkt_price, fees)


@pytest.mark.parametrize(
    "trade, expected_exec_info",
    [
        (long_trade, expected_exec_info["long"]),
        (short_trade, expected_exec_info["short"]),
    ],
)
def test_execute_trade(trade, expected_exec_info):
    # set up the context and the market
    ctx = {"trade_date": "2023-07-22"}
    cp_amm = ConstantProductEngine(
        mkt=MarketPair(
            pool_1=Pool("A", 2000),
            pool_2=Pool("B", 1000),
            swap_fee=0.01,
            mkt_price=2,
        )
    )

    # execute the trade
    result = cp_amm.execute_trade(ctx, trade)
    # check that all keys are present
    missing_keys = set(expected_exec_info.keys()) - set(result.keys())
    assert not missing_keys, f"Missing keys in the actual result: {missing_keys}"

    # check that all values are equal
    for key in expected_exec_info.keys():
        assert (
            expected_exec_info[key] == result[key]
        ), f"The value for key '{key}' does not match"
