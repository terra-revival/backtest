from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


class Pool:
    """A class that represents a liquidity pool."""

    def __init__(self, ticker: str, initial_deposit: float):
        """Initializes a new instance of the Pool class.

        Args:
            ticker (str): The ticker for the coin in the pool.
            initial_deposit (float): The initial deposit to the pool.

        """
        self.ticker = ticker
        self.reserves = [float(initial_deposit)]

    @property
    def balance(self) -> float:
        """Retrieves the current balance of the pool.

        Returns:
            float: The current balance of the pool.

        """
        assert len(self.reserves) > 0
        return self.reserves[-1]

    @property
    def initial_deposit(self) -> float:
        """Retrieves the initial deposit made to the pool.

        Returns:
            float: The initial deposit made to the pool.

        """
        assert len(self.reserves) > 0
        return self.reserves[0]


def split_ticker(trading_pair: str) -> Tuple[str, str]:
    """Splits a trading pair ticker into base and quote tickers.

    Args:
        trading_pair (str): The trading pair in the format 'BASE/QUOTE'.

    Returns:
        Tuple[str, str]: A tuple of the base and quote tickers.

    """
    assert trading_pair.count("/") == 1
    base_quote = trading_pair.split("/")
    return base_quote[0], base_quote[1]


class MarketQuote:
    """Represents a quote for the market price of a trading pair."""

    def __init__(self, trading_pair: str, price: float):
        """Initializes a new instance of the MarketQuote class.

        Args:
            trading_pair (str): The trading pair ticker, e.g., 'ETH/USD'.
            price (float): The market price of the trading pair.

        """
        base, quote = split_ticker(trading_pair)
        self.token_base = base
        self.token_quote = quote
        self.price = price

    @property
    def ticker(self) -> str:
        """Retrieves the ticker for the trading pair.

        Returns:
            str: The ticker for the trading pair in 'BASE/QUOTE' format.

        """
        return f"{self.token_base}/{self.token_quote}"

    def __str__(self) -> str:
        """Returns a string representation of the MarketQuote instance.

        Returns:
            str: The ticker and the price in 'BASE/QUOTE=price' format.

        """
        return f"{self.ticker}={self.price}"

    def __repr__(self) -> str:
        """Returns a developer-friendly, printable representation of the MarketQuote
        instance.

        Returns:
            str: The ticker and the price in 'BASE/QUOTE=price' format.

        """
        return f"{self.ticker}={self.price}"


class MarketPair:
    """A Market pair managing a liquidity pool made up of reserves of two tokens."""

    def __init__(self, pool_1: Pool, pool_2: Pool, swap_fee: float, mkt_price: float):
        """Initializes a new instance of the MarketPair class.

        Args:
            pool_1 (Pool): The first pool.
            pool_2 (Pool): The second pool.
            swap_fee (float): The swap fee.
            mkt_price (float): The market price.

        """
        # The ongoing reserves of the pool
        self.pool_1 = pool_1
        # The ongoing reserves of the pool
        self.pool_2 = pool_2
        # The swap fee
        self.swap_fee = swap_fee
        # The market price
        self.mkt_price_0 = mkt_price
        # The market price
        self.mkt_price = mkt_price
        # The market price
        self.total_fees_quote = float(0)
        # The market price
        self.volume_base = float(0)
        # The market price
        self.volume_quote = float(0)

    @property
    def ticker(self) -> str:
        """Ticker for the trading pair represented by this market."""
        return f"{self.pool_2.ticker}/{self.pool_1.ticker}"

    @property
    def inverse_ticker(self) -> str:
        """Ticker for the inverse trading pair represented by this market."""
        return f"{self.pool_1.ticker}/{self.pool_2.ticker}"

    @property
    def cp_invariant(self) -> float:
        """The constant product invariant."""
        return self.pool_1.balance * self.pool_2.balance

    @property
    def mid_price_0(self) -> float:
        """The initial mid price."""
        return self.pool_1.initial_deposit / self.pool_2.initial_deposit

    @property
    def mid_price(self) -> float:
        """The current mid price."""
        return self.pool_1.balance / self.pool_2.balance

    @property
    def spread(self) -> float:
        """The spread between the mid price and the market price."""
        return self.mkt_price - self.mid_price

    @property
    def avg_price(self) -> float:
        """The average execution price for all the trades."""
        return -self.volume_quote / self.volume_base

    @property
    def mkt_price_ratio(self) -> float:
        """The ratio of the current vs initial market price (moneyness)."""
        return self.mkt_price / self.mkt_price_0

    @property
    def impermanent_loss(self) -> float:
        """The current impermanent loss."""
        return (
            2.0 * (self.mkt_price_ratio**0.5) / (1.0 + self.mkt_price_ratio)
        ) - 1.0

    @property
    def start_base(self) -> float:
        """Initial reserve for the base currency."""
        return self.pool_2.balance - self.volume_base

    @property
    def start_quote(self) -> float:
        """Initial reserve for the quote currency."""
        return self.pool_1.balance - self.volume_quote

    @property
    def asset_base_0(self) -> float:
        """Initial reserve for the base currency expressed in quote currency."""
        return self.start_base * self.mkt_price_0

    @property
    def asset_base(self) -> float:
        """Current reserve for the base currency expressed in quote currency."""
        return self.pool_2.balance * self.mid_price

    @property
    def hodl_value(self) -> float:
        """Value of the hold portfolio eg.

        current market value of the initial reserves.

        """
        return self.start_quote + (self.start_base * self.mid_price)

    @property
    def value_0(self) -> float:
        """Initial value of the hold portfolio eg.

        market value of the initial reserves.

        """
        return self.start_quote + (self.start_base * self.mkt_price_0)

    @property
    def value(self) -> float:
        """Value of the LP portfolio eg.

        current market value of the current reserves.

        """
        return self.pool_1.balance + (self.pool_2.balance * self.mid_price)

    @property
    def trade_pnl(self) -> float:
        """Difference of value between the hold portfolio and the LP portfolio.

        This is the impermanent loss

        """
        return self.value - self.hodl_value

    @property
    def pnl(self) -> float:
        """Cash P&L of the LP position eg.

        total fees earned minus impermanent loss.

        """
        return self.trade_pnl + self.total_fees_quote

    @property
    def roi(self) -> float:
        """ROI of the LP portfolio."""
        return self.pnl / self.hodl_value

    def describe(self) -> Dict[str, float]:
        """Describes the market pair.

        Returns:
            Dict[str, float]: A dictionary describing the market pair.

        """
        return {
            "mid_price": self.mid_price,
            "mkt_price": self.mkt_price,
            "spread": self.spread,
            "avg_price": self.avg_price,
            "current_base": self.pool_2.balance,
            "current_quote": self.pool_1.balance,
            "cp_invariant": self.cp_invariant,
            "total_fees_paid_quote": self.total_fees_quote,
            "total_volume_base": self.volume_base,
            "total_volume_quote": self.volume_quote,
            "asset_base_pct": self.asset_base / self.value,
            "hold_portfolio": self.hodl_value,
            "current_portfolio": self.value,
            "trade_pnl": self.trade_pnl,
            "total_pnl": self.pnl,
            "roi": self.roi,
            "impermanent_loss": self.impermanent_loss,
            "mkt_price_ratio": self.mkt_price_ratio,
        }

    def assets(self) -> pd.DataFrame:
        """Calculates assets of the market pair.

        Returns:
            pd.DataFrame: A DataFrame describing assets of the market pair.

        """
        df = pd.DataFrame(
            index=[
                self.pool_2.ticker,
                self.pool_1.ticker,
                self.ticker,
                "pct_base_asset",
            ],
            data={
                "start": [
                    self.start_base,
                    self.start_quote,
                    self.mkt_price_0,
                    self.asset_base_0 / self.value_0,
                ],
                "current": [
                    self.pool_2.balance,
                    self.pool_1.balance,
                    self.mkt_price,
                    self.asset_base / self.value,
                ],
            },
        )
        df["change"] = df["current"] - df["start"]
        return df

    def perf(self) -> pd.DataFrame:
        """Calculates the performance of the market pair.

        Returns:
            pd.DataFrame: A DataFrame describing the performance of the market pair.

        """
        return pd.DataFrame(
            index=[
                "Hold Portfolio Value",
                "Current Portfolio Value",
                "Trade PnL",
                "Fees Paid",
                "Total PnL",
                "Return",
            ],
            data={
                "Values": [
                    self.hodl_value,
                    self.value,
                    self.trade_pnl,
                    self.total_fees_quote,
                    self.pnl,
                    self.roi,
                ]
            },
        )

    def get_delta_reserves(self) -> Tuple[float, float]:
        """The mid price of the trading pair."""
        mid_price = self.mid_price
        mkt_price = self.mkt_price
        sqrt_k = np.sqrt(self.cp_invariant)
        dx = sqrt_k * (np.sqrt(mkt_price) - np.sqrt(mid_price))
        dy = sqrt_k * (np.sqrt(1 / mid_price) - np.sqrt(1 / mkt_price))
        return dx, dy

    def get_reserves(self, trading_pair: str) -> Tuple[float, float]:
        """Returns reserves in correct order based on the trading pairs (normal or
        inversed).

        Args:
            trading_pair (str) :
                The trading pair ticker eg. ETH/USD

        Returns:
            Tuple[Pool, Pool]:
                (Liquidity pool 1, Liquidity pool 2)

        Raises:
            Exception: If the trading pair is unknown.

        """
        if trading_pair == self.ticker:
            return self.pool_1.balance, self.pool_2.balance
        elif trading_pair == self.inverse_ticker:
            return self.pool_2.balance, self.pool_1.balance
        else:
            raise Exception(f"Unknown trading pair {trading_pair}")

    def add_liquidity(
        self, liq_amount: float, quote_1: MarketQuote, quote_2: MarketQuote
    ):
        """Adds a given amount of liquidity in the AMM at the current price of the pool.
        The amount of token to add in each pool is determined from the market price of
        the tokens and the current price of the pool.

        Args:
            liq_amount (float) :
                The amount of liquidity to be added expressed
                in FIAT currency or token eg. USD, EUR, USDT, ETH etc.

            quote_1 (MarketQuote) :
                The market quote for the token in the first pool

            quote_2 (MarketQuote) :
                The market quote for the token in the second pool

        """
        x = self.pool_1.balance
        y = self.pool_2.balance
        alpha = (quote_1.price * x) / (quote_1.price * x + quote_2.price * y)
        liq_amount_1 = liq_amount * alpha / quote_1.price
        liq_amount_2 = liq_amount * (1 - alpha) / quote_2.price
        self.pool_1.reserves.append(x + liq_amount_1)
        self.pool_2.reserves.append(y + liq_amount_2)


def with_mkt_price(mkt, mkt_price):
    mkt.mkt_price = mkt_price
    return mkt


def new_market(
    liq_amount: float, quote_1: MarketQuote, quote_2: MarketQuote, swap_fee: float
) -> MarketPair:
    """Initializes a market with a given amount of liquidity and market prices for the
    tokens.

    Args:
        liq_amount (float) :
            The amount of liquidity to be added expressed
            in FIAT currency or token eg. USD, EUR, USDT, ETH etc.

        quote_1 (MarketQuote) :
            The market quote for the token in the first pool

        quote_2 (MarketQuote) :
            The market quote for the token in the second pool

        swap_fee (float) :
            The transaction fee per swap always paid in the base currency (token in)

    Returns:
        MarketPair:
            New market pair

    """
    liq_per_token = liq_amount / 2.0
    x_0 = liq_per_token / quote_1.price
    y_0 = liq_per_token / quote_2.price
    return MarketPair(
        Pool(quote_1.token_base, x_0),
        Pool(quote_2.token_base, y_0),
        swap_fee,
        quote_2.price / quote_1.price,
    )


class TradeOrder:
    """A trade order for a swap to execute."""

    def __init__(self, trading_pair: str, order_size: float, transaction_fees: float):
        self.ticker = trading_pair
        # the order size
        self.order_size = abs(order_size)
        # the direction of the order
        self.direction = "buy" if order_size > 0 else "sell"
        # the order size minus transaction fees
        self.net_order_size = self.order_size / (1.0 + transaction_fees)
        # the trannsaction fees
        self.cash_transaction_fee = transaction_fees * self.order_size

    @property
    def long(self) -> bool:
        return self.direction == "buy"

    @property
    def short(self) -> bool:
        return self.direction == "sell"

    @classmethod
    def create_default(cls, mkt: MarketPair) -> TradeOrder:
        """Default order equal to 10% of the first pool."""
        return cls(mkt.ticker, 0.1 * mkt.pool_1.balance, mkt.swap_fee)
