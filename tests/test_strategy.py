from pytest import approx
from datetime import datetime
from terra_algo_backtest.exec_engine import ConstantProductEngine
from terra_algo_backtest.market import MarketPair, Pool
from terra_algo_backtest.strategy import ArbStrategy, SimpleUniV2Strategy, DivStrategy

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


class TestDivStrategy:
    def setup_method(self):
        self.ctx = {
            "trade_date": "2023-07-22",
            "quantity": 1000,
            "price": 31_000,
        }
        div_protocol_args = {
            "soft_peg_price": 0.1,  # start right at the peg price, seems a smart decision
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
        self.strategy = DivStrategy(
            args=div_protocol_args,
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

    def test_calc_taxes(self):

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
            cex, chain = self.strategy.calc_taxes(
                test["price"], test["volume"])
            assert cex == test["output"][0]
            assert chain == test["output"][1]
