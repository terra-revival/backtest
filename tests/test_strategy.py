from pytest import approx

from terra_algo_backtest.exec_engine import ConstantProductEngine
from terra_algo_backtest.market import MarketPair, Pool
from terra_algo_backtest.strategy import ArbStrategy, SimpleUniV2Strategy

from .test_exec_engine import assert_exec_info


class TestArbStrategy:
    def setup_method(self):
        self.ctx = {
            "trade_date": "2023-07-22",
            "price": 31_000,
        }

        self.strategy = ArbStrategy(
            cp_amm=ConstantProductEngine(
                mkt=MarketPair(
                    pool_1=Pool("USD", 3_000_000),
                    pool_2=Pool("BTC", 100),
                    mkt_price=self.ctx["price"],
                    swap_fee=0,
                ),
            ),
        )

    def test_execute(self):
        trade_exec_info = self.strategy.execute(self.ctx, None)
        assert len(trade_exec_info) == 1

        assert_exec_info(
            trade_exec_info[0],
            {
                "trade_date": "2023-07-22",
                "side": "buy",
                "total_fees_paid_quote": 0.0,
                "mkt_price": approx(31_000),
                "mid_price": approx(31_000),
            },
        )


class TestSimpleUniV2Strategy:
    def setup_method(self):
        self.ctx = {
            "trade_date": "2023-07-22",
            "quantity": 1000,
            "price": 31_000,
        }

        self.strategy = SimpleUniV2Strategy(
            arb_enabled=True,
            cp_amm=ConstantProductEngine(
                mkt=MarketPair(
                    pool_1=Pool("USD", 3_000_000),
                    pool_2=Pool("BTC", 100),
                    mkt_price=self.ctx["price"],
                    swap_fee=0,
                ),
            ),
        )

    def test_execute(self):
        trade_exec_info = self.strategy.execute(self.ctx, None)
        assert len(trade_exec_info) == 2

        # arb
        assert_exec_info(
            trade_exec_info[0],
            {
                "trade_date": "2023-07-22",
                "side": "buy",
                "total_fees_paid_quote": 0.0,
                "mkt_price": approx(31_000),
                "mid_price": approx(31_000),
            },
        )

        assert_exec_info(
            trade_exec_info[1],
            {
                "trade_date": "2023-07-22",
                "side": "buy",
                "total_fees_paid_quote": 0.0,
                "mkt_price": approx(31_000),
            },
        )
