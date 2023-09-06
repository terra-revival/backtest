from abc import ABC, abstractmethod
from typing import Dict, Tuple

from .market import MarketPair, TradeOrder
from .swap import constant_product_swap, swap_price


class ExecEngine(ABC):
    @abstractmethod
    def execute_trade(self, ctx: Dict, order: TradeOrder) -> Dict:
        pass


class ConstantProductEngine(ExecEngine):
    def __init__(self, mkt: MarketPair):
        self.mkt = mkt

    def execute_trade(self, ctx: Dict, order: TradeOrder) -> Dict:
        """Executes a trade order using a MarketPair as the execution engine.

        Args:
            ctx (dict): The context in which the trade is being executed.
            order (TradeOrder): The trade order to execute.
            mkt (MarketPair): The MarketPair to use as the execution engine.

        Returns:
            dict: The result of the trade execution.

        """
        try:
            self.mkt.mkt_price = ctx["price"]

            mid_price = self.mkt.mid_price
            qty_received, exec_price = constant_product_swap(self.mkt, order)
            volume_base = qty_received if order.long else order.order_size
            volume_quote = order.order_size if order.long else qty_received
            price_impact = (exec_price / mid_price) - 1

            return {
                "trade_date": ctx["trade_date"],
                "side": order.direction,
                "price": exec_price,
                "price_impact": price_impact,
                "trade_qty": order.order_size,
                "qty_received": qty_received,
                "volume_base": volume_base,
                "volume_quote": volume_quote,
                **self.mkt.describe(),
            }
        except AttributeError as e:
            raise AttributeError(
                "Invalid attribute in one of the input objects: " + str(e)
            )
        except KeyError as e:
            raise KeyError("Missing key in context dictionary: " + str(e))

    def get_exec_price(self, size: float) -> Tuple[float, float]:
        """Gets the execution price for a trade order.

        Args:
            ctx (dict): The context in which the trade is being executed.
            order (TradeOrder): The trade order to execute.

        Returns:
            Tuple[float, float]: The execution price and the price impact.

        """
        try:
            exec_price, mid_price = swap_price(
                self.mkt, TradeOrder(size, self.mkt.swap_fee)
            )
            return exec_price, mid_price
        except AttributeError as e:
            raise AttributeError(
                "Invalid attribute in one of the input objects: " + str(e)
            )
        except KeyError as e:
            raise KeyError("Missing key in context dictionary: " + str(e))

    def calc_arb_trade(self, target_price: float) -> Tuple[TradeOrder | None, float]:
        """Calculates the trade order that would be executed to arb the DEX against
        another venue."""
        dx, dy = self.mkt.get_delta_reserves(target_price)
        if dy == 0 or dx == 0:
            return None, 0.0

        qty = dx if dx > 0 else dy
        trade = TradeOrder(qty, self.mkt.swap_fee)
        return trade, dx / dy


def calc_arb_trade_pnl(
    trade: TradeOrder, pool_exec_price: float, mkt_price: float, fees: float
) -> float:
    """Calculates the profit and loss for a trade between a DEX and another venue.

    Args:
        trade (TradeOrder): The trade order.
        pool_exec_price (float): The execution price for the trade against the DEX.
        mkt_price (float): The current market price for the pair from another venue.
        fees (float): The total fees associated with the trade.
        Should be the sum of the fees paid to the DEX and the other venue plus any
        other fees such as the blockchain tx fees for example.

    Returns:
        float: The calculated profit and loss.

    """
    try:
        if trade.long:
            spread = (mkt_price - pool_exec_price) / pool_exec_price
        else:
            spread = (pool_exec_price - mkt_price) / mkt_price
        return (spread - fees) * trade.order_size
    except ZeroDivisionError:
        raise ValueError("The price cannot be zero.")
    except AttributeError:
        raise TypeError("Invalid input, 'trade' must be an instance of TradeOrder.")
