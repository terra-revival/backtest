import pytest
from pytest import approx

from terra_algo_backtest.exec_engine import ConstantProductEngine
from terra_algo_backtest.market import MarketPair, Pool
from terra_algo_backtest.strategy import (
    ArbStrategy,
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
        (1.0, 1.0, 1.0, -1000),
        (1.0, 1.0, 1.0, -10000),
        (1.0, 1.0, 1.0, -100),
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
            peg_price=peg_price,
            strategy=SimpleUniV2Strategy(
                arb_enabled=True,
                cp_amm=self.cp_amm,
            ),
        )

    def test_execute(self):
        # Scenario 1: Price is below Peg of $1USD (X>Y)
        trade_exec_info = self.strategy.execute(self.ctx, None)

        assert len(trade_exec_info) == 1

        exec_info = trade_exec_info[0]
        if trade_exec_info[0]["side"] == "buy":
            assert "div_tax_pct" not in exec_info

        exec_price = exec_info["price"]
        mid_price = exec_info["mid_price"]
        qty_received = exec_info["qty_received"]

        peg_price = self.strategy.peg_price
        expected_div_tax_pct = 1 - (mid_price / peg_price)
        expected_div_tax = qty_received * expected_div_tax_pct / exec_price

        dx, _ = self.cp_amm.mkt.get_delta_reserves(self.strategy.peg_price)
        expected_buy_back_qty = min(dx, expected_div_tax)
        expected_bb_exec_price, expected_bb_mid_price = self.cp_amm.get_exec_price(
            "buy", expected_buy_back_qty
        )

        # Transformed code
        assert exec_info["trade_date"] == "2023-07-22"
        assert exec_info["total_fees_paid_quote"] == 0.0
        assert exec_info["div_tax_pct"] == expected_div_tax_pct
        assert exec_info["div_tax_quote"] == expected_div_tax
        assert exec_info["adj_qty_received"] == qty_received * (
            1.0 - expected_div_tax_pct
        )
        assert exec_info["buy_back_price"] == expected_bb_exec_price
        assert exec_info["buy_back_mid_price"] == expected_bb_mid_price
        assert exec_info["mid_price"] == mid_price
        assert exec_info["price"] == exec_price

        import pprint

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(trade_exec_info[0])


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
            peg_price=peg_price,
            strategy=SimpleUniV2Strategy(
                arb_enabled=True,
                cp_amm=self.cp_amm,
            ),
        )

    def test_calc_div_tax(self):
        trade_exec_info = self.strategy.execute(self.ctx, None)
        import pprint

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(trade_exec_info[0])

        for exec_info in trade_exec_info:
            # Scenario 1: Price is below Peg of $1USD (X>Y)
            tax = self.strategy.calc_div_tax(exec_info)
            if exec_info["side"] == "buy":
                assert tax == 0.0
            else:
                assert tax == approx(
                    1 - (exec_info["mid_price"] / self.strategy.peg_price)
                )
