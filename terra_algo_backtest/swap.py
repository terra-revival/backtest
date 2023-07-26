from math import sqrt
from typing import Tuple

import numpy as np
from loguru import logger

from .market import MarketPair, TradeOrder


class MidPrice:
    def __init__(self, trading_pair: str, x: float, y: float):
        """The price between what users can buy and sell tokens at a given moment. In
        Uniswap this is the ratio of the two ERC20 token reserves.

        Args:
            trading_pair (str) :
                The trading pair ticker eg. ETH/USD

            x (float) :
                Reserve of token A before swap

            y (float) :
                Reserve of token B before swap

        """
        assert x > 0
        assert y > 0
        assert trading_pair.count("/") == 1
        xy_ticker = trading_pair.split("/")
        self.x_ticker = xy_ticker[0]
        self.y_ticker = xy_ticker[1]
        self.x = x
        self.y = y
        self.mid_price = x / y


class PriceImpactRange:
    def __init__(self, start: MidPrice, mid: MidPrice, end: MidPrice):
        """Price impact range of a swap. Starts at mid price before the swap, ends at mid
        price after the swap. Also contains the point where mid price is equal to the
        swap execution price.

        Args:
            start (MidPrice) :
                point corresponding to the mid price of the pool before swap

            mid (MidPrice) :
                point corresponding to the execution price of the swap
                eg. (x,y,x/y) where mid price == execution price

            end (MidPrice) :
                point corresponding to the next mid price of the pool
                eg. next price after swap

        """
        self.start = start
        self.mid = mid
        self.end = end


def assert_cp_invariant(x: float, y: float, k: float, precision: float | None = None):
    """Asserts that the constant product is invariant.

    Args:
        x (float) :
            Reserve of tokens A

        y (float) :
            Reserve of tokens B

        k (float) :
            Constant product invariant

        precision (float) :
            Precision at which the constant product invariant is evaluated

    Returns:
        None

    """
    precision = precision or 1e-14
    try:
        assert k > 0
        assert abs(k - (x * y)) / k <= precision
    except Exception as e:
        logger.error("Constant product invariant not satisfied")
        logger.error(f"diff={abs((x*y) - k)}")
        logger.error(f"precision={precision}")
        logger.error(f"x={x}")
        logger.error(f"y={y}")
        logger.error(f"x*y={x*y}")
        logger.error(f"k={k}")
        raise e


def constant_product_swap(
    mkt: MarketPair,
    order: TradeOrder,
    precision: float | None = None,
) -> Tuple[float, float]:
    """Swap tokens A for tokens B from pool with a XY constant product.

    Args:
        mkt (MarketPair) :
            The market pair to trade against

        order (TradeOrder) :
            The trade order to execute

        precision (float) :
            Precision at which constant product invariant is evaluated

    Returns:
        Tuple[float, float] :
            (Amount of tokens B out, Swap execution price)

    """
    assert order.order_size != 0
    # the reserves depending on the swap direction
    x, y = mkt.pool_1.balance, mkt.pool_2.balance
    # the order size
    if order.direction == "buy":
        dx = order.net_order_size
        # calculate dy amount of tokens B to be taken out from the AMM
        dy = (y * dx) / (x + dx)
        # add dx amount of tokens A to the AMM
        mkt.pool_1.reserves.append(x + dx)
        # take dy amount of tokens B out from the AMM
        mkt.pool_2.reserves.append(y - dy)
        mkt.volume_base -= dy
        mkt.volume_quote += dx / (1 - mkt.swap_fee)
        mkt.total_fees_quote += mkt.swap_fee * dx / (1 - mkt.swap_fee)
        # assert k is still invariant
        assert_cp_invariant(
            mkt.pool_1.balance, mkt.pool_2.balance, mkt.cp_invariant, precision
        )
        return dy, dx / dy
    elif order.direction == "sell":
        dy = order.net_order_size
        # calculate dx amount of tokens A to be taken out from the AMM
        dx = (x * dy) / (y + dy)
        # add dx amount of tokens A to the AMM
        mkt.pool_1.reserves.append(x - dx)
        # take dy amount of tokens B out from the AMM
        mkt.pool_2.reserves.append(y + dy)
        mkt.volume_base += dy
        mkt.volume_quote -= dx / (1 - mkt.swap_fee)
        mkt.total_fees_quote += mkt.swap_fee * dx / (1 - mkt.swap_fee)
        # assert k is still invariant
        assert_cp_invariant(
            mkt.pool_1.balance, mkt.pool_2.balance, mkt.cp_invariant, precision
        )
        return dx, dx / dy
    else:
        raise Exception(
            f"Trade order direction must be buy or sell. Got {order.direction}"
        )


def swap_price(x, y, dx) -> float:
    """Computes the swap execution price for an order size given two pools with reserves
    x and y.

    Args:
        x (float) :
            Reserve of tokens A

        y (float) :
            Reserve of tokens B

        dx (float) :
            Order size

    Returns:
        float :
            Swap execution price

    """
    return (x + dx) / y


def constant_product_curve(
    mkt: MarketPair,
    x_min: float | None = None,
    x_max: float | None = None,
    num: int | None = None,
) -> Tuple[list[float], list[float]]:
    """Computes the AMM curve Y = K/X for a constant product AMM K = XY

    Args:
        mkt (MarketPair) :
            The market pair to trade against

        x_min (float) :
            minimum value of X

        x_max (float) :
            maximum value of X

        num (float) :
            number of points to be computed

    Returns:
        Tuple[list[float],list[float]] :
            (Amount of tokens B out, Swap execution price)

    """
    num = num or 1000
    x_min = x_min or 0.1 * mkt.pool_1.balance
    x_max = x_max or 5.0 * mkt.pool_1.balance
    x = np.linspace(x_min, x_max, num=num)
    y = mkt.cp_invariant / x
    return x, y


def price_impact_range(
    mkt: MarketPair,
    order: TradeOrder | None = None,
    precision: float | None = None,
) -> PriceImpactRange:
    """Price impact of a trade order against a market.

    Args:
        mkt (MarketPair) :
            The market pair to trade against

        order (TradeOrder) :
            The trade order to execute

        precision (float) :
            precision at which the invariant is evaluated

    Returns:
        PriceImpactRange :
            Price impact range for given pools and order size

    """
    # create default trade order
    order = order or TradeOrder.create_default(mkt)
    # trade size provided or defaulted to 10% of x
    dx = order.order_size
    # constant product invariant
    k = mkt.cp_invariant
    # start: (x,y)
    x_start, y_start = mkt.get_reserves(order.ticker)
    # end: (x+dx, y-dy)
    x_end = x_start + dx
    y_end = y_start * (1.0 - dx / (x_start + dx))
    # assert k is invariant at start and end
    assert_cp_invariant(x_start, y_start, k, precision)
    assert_cp_invariant(x_end, y_end, k, precision)
    # swap execution price at start for dx amount of tokens A
    exec_price = swap_price(x_start, y_start, dx)
    # (x, y) of the mid price equal to the execution price
    x_mid = sqrt(k * exec_price)
    y_mid = k / sqrt(k * exec_price)
    return PriceImpactRange(
        MidPrice(mkt.ticker, x_start, y_start),
        MidPrice(mkt.ticker, x_mid, y_mid),
        MidPrice(mkt.ticker, x_end, y_end),
    )


def order_book(
    mkt: MarketPair,
    x_min: float | None = None,
    x_max: float | None = None,
    num: int | None = None,
):
    """Computes the cumulative quantity at any mid price according to the formula from
    the paper "Order Book Depth and Liquidity Provision in Automated Market Makers".

    Args:
        mkt (MarketPair) :
            The market pair to trade against

        x_min (float) :
            minimum value of X

        x_max (float) :
            maximum value of X

        num (float) :
            number of points to be computed

    Returns:
        Tuple[list[float],list[float]] :
            (reserves of token A, reserves of token B)

    """
    q = []
    x_0 = float(mkt.pool_1.initial_deposit)
    p_0 = float(mkt.pool_1.initial_deposit / mkt.pool_2.initial_deposit)
    x, y = constant_product_curve(mkt, x_min, x_max, num)
    p = [x_i / y_i for x_i, y_i in zip(x, y)]
    for p_i in p:
        q_i = float(0)
        if p_i < p_0:
            q_i = x_0 * (sqrt(p_0 / p_i) - 1)
        if p_i > p_0:
            q_i = x_0 * (1 - sqrt(p_0 / p_i))
        q.append(q_i)
    return x, p, q
