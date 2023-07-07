# swap_simulation function updated with additional arguments for strategy
from typing import Callable, List

from .market import MarketPair, TradeOrder
from .swap import calc_arb_trade, constant_product_swap


def get_strategy(strategy: str) -> Callable[[dict, MarketPair], List[dict]]:
    """Returns the strategy functon.

    Args:
        strategy (str): the strategy name.

    Returns:
        Callable[[dict, MarketPair], List[dict]]: strategy function.

    """
    if strategy == "uni_v2":
        return uni_v2
    if strategy == "div_protocol":
        return div_protocol
    raise Exception(f"Strategy {strategy} not found")


def uni_v2(row: dict, mkt: MarketPair) -> List[dict]:
    """UNI v2 LP strategy.

    Args:
        row (dict): The trade data.
        mkt (MarketPair): The market pair for which swaps are to be simulated.

    Returns:
        dict: trade_exec_info.

    """

    trade_exec_info = []
    quantity, pnl = calc_arb_trade(mkt)
    if pnl > 0:  # only execute if profitable
        trade_exec_info.append(execute_trade(mkt, row["trade_date"], quantity, pnl))
    if row["quantity"] != 0:
        trade_exec_info.append(execute_trade(mkt, row["trade_date"], row["quantity"]))
    return trade_exec_info


def div_protocol(row: dict, mkt: MarketPair) -> List[dict]:
    """UNI v2 LP strategy with divergence protocol.

    Args:
        row (dict): The trade data.
        mkt (MarketPair): The market pair for which swaps are to be simulated.

    Returns:
        dict: trade_exec_info.

    """
    trade_exec_info = uni_v2(row, mkt)
    # divergence tax if applicable
    return trade_exec_info


def execute_trade(
    mkt: MarketPair, trade_date: object, volume: float, arb_profit: float = 0
) -> dict:
    """Executes a trade for a given market pair and volume.

    Args:
        mkt (MarketPair): The market pair for which the trade is to be executed.
        trade_date (object): The date of the trade.
        volume (object): The volume of the trade.
        arb_profit (float, optional): The profit from arbitrage. Defaults to 0.

    Returns:
        dict: A dictionary with information about the executed trade.

    """
    mid_price = mkt.mid_price
    trade = TradeOrder(mkt.ticker, volume, mkt.swap_fee)
    _, exec_price = constant_product_swap(mkt, trade)
    # _, exec_price = mock_constant_product_swap(mkt, trade)
    return {
        "trade_date": trade_date,
        "side": trade.direction,
        "arb_profit": arb_profit,
        "price": exec_price,
        "price_impact": (mid_price - exec_price) / mid_price,
        **mkt.describe(),
    }
