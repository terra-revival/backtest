import random

import numpy as np
from pytest import approx

from terra_algo_backtest.market import MarketPair, Pool, TradeOrder
from terra_algo_backtest.swap import constant_product_curve, constant_product_swap


class TestConstantProductSwapSimple:
    """This class contains a simple unit tests for the `constant_product_swap` function.

    It corresponds to the famous uniswap v2 example illustrated in the image below:
    https://docs.uniswap.org/assets/images/trade-b19a05be2c43a62708ab498766dc6d13.jpg

    """

    def setup_method(self):
        self.market_pair = MarketPair(
            pool_2=Pool("A", 400),
            pool_1=Pool("B", 1200),
            swap_fee=0.003,
            mkt_price=3.0,
        )

        self.trade_order = TradeOrder(
            order_size=3.0,
            transaction_fees=self.market_pair.swap_fee,
        )

    def test_constant_product_swap(self):
        # Check initial conditions
        assert self.market_pair.pool_1.balance == approx(1200)
        assert self.market_pair.pool_2.balance == approx(400)
        assert self.market_pair.cp_invariant == approx(1200 * 400)
        assert self.market_pair.mid_price == approx(3.0)
        assert self.market_pair.mid_price_0 == approx(3.0)

        # swap
        qty_received, exec_price = constant_product_swap(
            self.market_pair, self.trade_order
        )

        # Check final conditions
        assert self.market_pair.pool_1.balance == approx(1203.009)
        assert self.market_pair.pool_2.balance == approx(399.003)
        assert self.market_pair.cp_invariant == approx(1203.009 * 399.003)
        assert self.market_pair.mid_price == approx(3.015)
        assert self.market_pair.mid_price_0 == approx(3.0)
        assert qty_received == approx(0.997)


class TestConstantProductSwap:
    """This class contains a unit tests for the `constant_product_swap` function.

    - Asset base = USD, starting balance = 300_000 USD
    - Asset quote = BTC, starting balance = 10 BTC
    - Swap fee = 1%, Mid price = 30,000 USD/BTC

    We generate an XY curve for the market pair to check the results
    - X = USD balance, Y = BTC balance
    - Start X = 1 USD, End X = 6,000,000 USD
    - Discretization steps = 1,000,000

    We then proceed to swap back and forth between the two assets
    - We toss a coin to decide whether to buy or sell, head = buy, tail = sell
    - We randomly generate the order size between 100 USD and 45,000 USD for long orders
    - We randomly generate the order size between 0.01 BTC and 2 BTC for short orders
    - We check results against the discretized XY curve

    """

    def setup_method(self):
        self.market_pair = MarketPair(
            pool_1=Pool("USD", 300_000),
            pool_2=Pool("BTC", 10),
            swap_fee=0.01,
            mkt_price=30_000,
        )

        self.x = np.linspace(start=1, stop=6_000_000, num=1_000_000)
        self.y = self.market_pair.cp_invariant / self.x

    def test_constant_product_swap(self):
        x_i = self.market_pair.pool_1.balance
        y_i = np.interp(x_i, self.x, self.y)

        for _ in range(1, 10000):
            prev_x, prev_y = x_i, y_i

            if random.random() < 0.5:
                order = TradeOrder(
                    random.uniform(100, 45000), self.market_pair.swap_fee
                )
            else:
                order = TradeOrder(-random.uniform(0.01, 2), self.market_pair.swap_fee)

            qty_received, exec_price = constant_product_swap(self.market_pair, order)
            dx = abs(order.net_order_size) if order.long else abs(qty_received)
            dy = abs(qty_received) if order.long else abs(order.net_order_size)

            x_i = self.market_pair.pool_1.balance
            y_i = np.interp(x_i, self.x, self.y)

            assert abs(x_i - prev_x) == approx(dx)
            assert abs(y_i - prev_y) == approx(dy)
            assert exec_price == approx(abs(dx / dy))
            assert self.market_pair.mid_price == approx(x_i / y_i)


class TestConstantProductCurve:
    """This class contains a simple unit tests for the `constant_product_curve` function.

    We generate an XY curve for the market pair using the
    `constant_product_curve` function. We then proceed to check each
    point on the curve by swapping between the two assets.

    """

    def setup_method(self):
        self.market_pair = MarketPair(
            pool_1=Pool("USD", 3_000_000),
            pool_2=Pool("BTC", 100),
            swap_fee=0,
            mkt_price=30_000,
        )

    def test_constant_product_curve(self):
        x, y = constant_product_curve(
            self.market_pair,
            x_min=0.01 * self.market_pair.pool_1.initial_deposit,
            x_max=10.0 * self.market_pair.pool_1.initial_deposit,
            num=10_000,
        )

        mkt = MarketPair(
            pool_1=Pool("USD", x[0]),
            pool_2=Pool("BTC", y[0]),
            swap_fee=self.market_pair.swap_fee,
            mkt_price=self.market_pair.mkt_price,
        )

        x_actual, y_actual = [x[0]], [y[0]]
        for dx in np.diff(x):
            order = TradeOrder(dx, mkt.swap_fee)
            dy, _ = constant_product_swap(mkt, order)
            x_actual.append(x_actual[-1] + dx)
            y_actual.append(y_actual[-1] - dy)

        np.testing.assert_allclose(x_actual, x, rtol=1e-14, equal_nan=True)
        np.testing.assert_allclose(y_actual, y, rtol=1e-14, equal_nan=True)
