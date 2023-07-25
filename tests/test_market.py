from pytest import approx

from terra_algo_backtest.market import MarketPair, MarketQuote, Pool, TradeOrder
from terra_algo_backtest.swap import constant_product_swap


class TestPool:
    """This class contains unit tests for the `Pool` class."""

    def setup_method(self):
        self.pool = Pool("BTC", 1000.0)

    def test_ticker(self):
        assert self.pool.ticker == "BTC"

    def test_initial_deposit(self):
        assert self.pool.initial_deposit == 1000.0

    def test_balance(self):
        assert self.pool.balance == 1000.0


class TestMarketQuote:
    """This class contains unit tests for the `MarketQuote` class."""

    def setup_method(self):
        self.market_quote = MarketQuote("BTC/USD", 50000.0)

    def test_ticker(self):
        assert self.market_quote.ticker == "BTC/USD"

    def test_price(self):
        assert self.market_quote.price == 50000.0

    def test_str(self):
        assert str(self.market_quote) == "BTC/USD=50000.0"

    def test_repr(self):
        assert repr(self.market_quote) == "BTC/USD=50000.0"


class TestMarketPair:
    """This class contains unit tests for the `MarketPair` class."""

    def setup_method(self):
        pool_base = Pool("BTC", 100)
        pool_quote = Pool("USD", 3_000_000)
        self.market_pair = MarketPair(
            pool_quote, pool_base, swap_fee=0.01, mkt_price=31000
        )

    def test_ticker(self):
        assert self.market_pair.ticker == "BTC/USD"

    def test_inverse_ticker(self):
        assert self.market_pair.inverse_ticker == "USD/BTC"

    def test_cp_invariant(self):
        assert self.market_pair.cp_invariant == 300_000_000

    def test_mid_price_0(self):
        assert self.market_pair.mid_price_0 == 30000

    def test_mid_price(self):
        assert self.market_pair.mid_price == 30000

    def test_spread(self):
        assert self.market_pair.spread == 1000

    def test_avg_price(self):
        assert self.market_pair.avg_price == 0

    def test_mkt_price_ratio(self):
        assert self.market_pair.mkt_price_ratio == 1

    def test_impermanent_loss(self):
        assert self.market_pair.impermanent_loss == 0

    def test_start_base(self):
        assert self.market_pair.start_base == 100

    def test_start_quote(self):
        assert self.market_pair.start_quote == 3_000_000

    def test_asset_base_0(self):
        assert self.market_pair.asset_base_0 == 3_100_000

    def test_asset_base(self):
        assert self.market_pair.asset_base == 3_000_000

    def test_hodl_value(self):
        assert self.market_pair.hodl_value == 6_000_000

    def test_value_0(self):
        # start_quote + (start_base * mkt_price_0)
        # value_0 = 3_000_000 + (100 * 31000)
        assert self.market_pair.value_0 == 6_100_000

    def test_value(self):
        # quote + (base * mid_price)
        # value = 3_000_000 + (100 * 30000)
        assert self.market_pair.value == 6_000_000

    def test_trade_pnl(self):
        assert self.market_pair.trade_pnl == 0.0

    def test_pnl(self):
        assert self.market_pair.pnl == 0.0

    def test_roi(self):
        assert self.market_pair.roi == 0.0

    def test_describe(self):
        expected_description = {
            "mid_price": 30000,
            "mkt_price": 31000,
            "spread": 1000,
            "avg_price": 0.0,
            "current_base": 100,
            "current_quote": 3_000_000,
            "cp_invariant": 300_000_000,
            "total_fees_paid_quote": 0.0,
            "total_volume_base": 0.0,
            "total_volume_quote": 0.0,
            "asset_base_pct": 0.5,
            "hold_portfolio": 6_000_000,
            "current_portfolio": 6_000_000,
            "trade_pnl": 0.0,
            "total_pnl": 0.0,
            "roi": 0.0,
            "impermanent_loss": 0.0,
            "mkt_price_ratio": 1.0,
        }
        assert self.market_pair.describe() == expected_description


class TestMarketPairSwap:
    """This class contains unit tests for the `MarketPair` class."""

    def setup_method(self):
        pool_base = Pool("BTC", 100)
        pool_quote = Pool("USD", 3_000_000)
        self.market_pair = MarketPair(
            pool_quote, pool_base, swap_fee=0.01, mkt_price=31000
        )
        self.qty, self.exec_price = constant_product_swap(
            self.market_pair,
            TradeOrder("BTC/USD", 100, 0.01),
        )

    def test_mid_price_0(self):
        assert self.market_pair.mid_price_0 == 30000

    def test_start_base(self):
        assert self.market_pair.start_base == 100

    def test_start_quote(self):
        assert self.market_pair.start_quote == approx(2_999_998.999)

    def test_asset_base_0(self):
        assert self.market_pair.asset_base_0 == 3_100_000

    def test_asset_base(self):
        assert self.market_pair.asset_base == approx(3_000_099.009)

    def test_value_0(self):
        # start_quote + (start_base * mkt_price_0)
        # value_0 = 3_000_000 + (100 * 31000)
        assert self.market_pair.value_0 == approx(6_099_998.999)

    def test_describe(self):
        expected_description = {
            "mid_price": 30001.980230696336,
            "mkt_price": 31000,
            "spread": 998.0197693036644,
            "avg_price": 30304.030403040306,
            "current_base": 99.99669977888519,
            "current_quote": 3_000_099.00990099,
            "cp_invariant": 300_000_000,
            "total_fees_paid_quote": 1.0001000100010002,
            "total_volume_base": -0.003300221114814693,
            "total_volume_quote": 100.01000100010002,
            "asset_base_pct": 0.5,
            "hold_portfolio": 6_000_197.022969624,
            "current_portfolio": 6_000_198.01980198,
            "trade_pnl": 0.9968323558568954,
            "total_pnl": 1.9969323658578957,
            "roi": 3.328111324033776e-07,
            "impermanent_loss": 0.0,
            "mkt_price_ratio": 1.0,
        }

        assert self.market_pair.describe() == expected_description
