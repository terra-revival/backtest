import pytest
from pytest import approx

from terra_algo_backtest.exec_engine import ConstantProductEngine
from terra_algo_backtest.market import MarketPair, Pool
from terra_algo_backtest.strategy import (
    ArbStrategy,
    DivProtocolParams,
    DivProtocolStrategy,
    SimpleUniV2Strategy,
)

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


@pytest.mark.parametrize(
    "peg_price, mkt_price, mid_price, order_size",
    [
        (1.0, 1.0, 1.0, -100),
    ],
)
class TestDivProtocolStrategy:
    @pytest.fixture(scope="function", autouse=True)
    def setup_method(self, peg_price, mkt_price, mid_price, order_size):
        self.ctx = {
            "trade_date": "2023-07-22",
            "quantity": order_size,
            "price": mkt_price,
        }

        self.cp_amm = ConstantProductEngine(
            mkt=MarketPair(
                pool_1=Pool("BUSD", 1000),
                pool_2=Pool("USTC", 1000 * mid_price),
                mkt_price=mkt_price,
                swap_fee=0,
            ),
        )

        self.strategy = DivProtocolStrategy(
            strategy=SimpleUniV2Strategy(
                arb_enabled=True,
                cp_amm=self.cp_amm,
            ),
            strat_params=DivProtocolParams(
                peg_price=peg_price,
                pct_buy_back=1.0,
            ),
        )

        self.reserve_account = 0.0

    def test_execute(self):
        # Scenario 1: Price is below Peg of $1USD (X>Y)
        trade_exec_info = self.strategy.execute(self.ctx, None)
        peg_price = self.strategy.params.peg_price
        for exec_info in trade_exec_info:
            mid_price = exec_info["mid_price"]
            volume_base = exec_info["volume_base"]
            volume_quote = exec_info["volume_quote"]

            expected_div_tax_pct = abs(1 - (mid_price / peg_price))
            expected_div_tax_quote = volume_quote * expected_div_tax_pct
            expected_div_volume_quote = volume_quote * (1 - expected_div_tax_pct)
            expected_div_exec_price = expected_div_volume_quote / volume_base

            if "buy_back_volume_quote" not in exec_info:
                assert exec_info is not None
                self.reserve_account += exec_info["div_tax_quote"]

                assert exec_info["div_tax_pct"] == approx(expected_div_tax_pct)
                assert exec_info["div_tax_quote"] == approx(expected_div_tax_quote)
                assert exec_info["div_volume_quote"] == approx(
                    expected_div_volume_quote
                )
                assert exec_info["div_exec_price"] == approx(expected_div_exec_price)
                assert exec_info["div_tax_quote"] == approx(expected_div_tax_quote)
            else:
                # calculate buy back volume
                dx, _ = self.cp_amm.mkt.get_delta_reserves(peg_price)
                buy_back_volume_quote = min(self.reserve_account, dx)
                self.reserve_account -= buy_back_volume_quote

                assert exec_info["reserve_account"] == self.reserve_account
                assert exec_info["buy_back_volume_quote"] == buy_back_volume_quote


@pytest.mark.parametrize(
    "peg_price, order_size",
    [
        (0.1, -10),
    ],
)
class TestDivTax:
    @pytest.fixture(scope="function", autouse=True)
    def setup_method(self, peg_price, order_size):
        self.ctx = {
            "trade_date": "2023-07-22",
            "quantity": order_size,
            "price": peg_price,
        }

        self.cp_amm = ConstantProductEngine(
            mkt=MarketPair(
                pool_1=Pool("BUSD", 1000),
                pool_2=Pool("USTC", 1000 / peg_price),
                mkt_price=peg_price,
                swap_fee=0,
            ),
        )

        self.strategy = DivProtocolStrategy(
            strategy=SimpleUniV2Strategy(
                arb_enabled=True,
                cp_amm=self.cp_amm,
            ),
            strat_params=DivProtocolParams(
                peg_price=peg_price,
                pct_buy_back=1.0,
            ),
        )

    def test_calc_div_tax(self):
        trade_exec_info = self.strategy.execute(self.ctx, None)
        peg_price = self.strategy.params.peg_price
        for exec_info in trade_exec_info:
            mid_price = exec_info["mid_price"]
            volume_base = exec_info["volume_base"]
            volume_quote = exec_info["volume_quote"]

            expected_div_tax_pct = abs(1 - (mid_price / peg_price))
            expected_div_tax_quote = volume_quote * expected_div_tax_pct
            expected_div_volume_quote = volume_quote * (1 - expected_div_tax_pct)
            expected_div_exec_price = expected_div_volume_quote / volume_base

            div_exec_info = self.strategy.calc_div_tax(exec_info)

            assert div_exec_info is not None
            assert div_exec_info["div_tax_pct"] == approx(expected_div_tax_pct)
            assert div_exec_info["div_tax_quote"] == approx(expected_div_tax_quote)
            assert div_exec_info["div_volume_quote"] == approx(
                expected_div_volume_quote
            )
            assert div_exec_info["div_exec_price"] == approx(expected_div_exec_price)
