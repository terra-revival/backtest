# swap_simulation function updated with additional arguments for strategy
from typing import Callable, List
import datetime
from market import MarketPair, TradeOrder
from swap import calc_arb_trade, constant_product_swap, price_impact_range


def get_strategy(strategy: str) -> Callable[[dict, MarketPair, dict], List[dict]]:
    """Returns the strategy functon.

    Args:
        strategy (str): the strategy name.

    Returns:
        Callable[[dict, MarketPair], List[dict]]: strategy function.

    """
    if strategy == "uni_v2":
        return uni_v2
    raise Exception(f"Strategy {strategy} not found")


def uni_v2(row: dict, mkt: MarketPair, args: dict) -> List[dict]:
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


def div_protocol(trade: TradeOrder, mkt: MarketPair, args: dict, dt: datetime.datetime, is_buy_back: bool = False) -> List[dict]:
    """UNI v2 LP strategy with divergence protocol.

    Args:
        row (dict): The trade data.
        mkt (MarketPair): The market pair for which swaps are to be simulated.

    Returns:
        dict: trade_exec_info.

    """
    trade_exec_info = []
    mid_price = mkt.mid_price
    _, exec_price = constant_product_swap(mkt, trade)
    executed = {
        "trade_date": dt,
        "side": trade.direction,
        "arb_profit": 0,
        "price": exec_price,
        "price_impact": (mid_price - exec_price) / mid_price,
        **mkt.describe(),
    }

    # divergence tax if applicable
    if is_buy_back:
        executed['buy_back_vol'] = trade.order_size
    else:
        if (exec_price < args['soft_peg_price'] and args['soft_peg_price'] == 1) or (args['soft_peg_price'] < 1):
            T = abs(exec_price - args['soft_peg_price']) * (1 - args['arbitrage_coef'])
            executed['cex_profit'] = T * args['cex_tax_coef'] * trade.order_size
            executed['chain_profit'] = T * (1 - args['cex_tax_coef']) * trade.order_size
    trade_exec_info.append(executed)
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
