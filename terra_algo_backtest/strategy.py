# swap_simulation function updated with additional arguments for strategy
import datetime
from typing import Callable, List, Tuple

from .market import MarketPair, TradeOrder
from .swap import calc_arb_trade, constant_product_swap


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


def calc_taxes(exec_price: float, volume: float, args: dict) -> Tuple[float, float]:
    """calculates div protocol for a trade and returns tax in USDT.

    Args:
        exec_price (float): execution price of the trade
        peg_price (float): current peg price
        volume (float): volume of trade, buy > 0, sell < 0.
                        Note, sell volume is in quote, buy in base
        args (dict): div protocol constants

    Returns:
        Tuple[float, float]: cex and chain profit

    """
    if exec_price < args["soft_peg_price"] or (
        args["soft_peg_price"] == 1 and exec_price > 1
    ):
        # until we are at peg = 1, we tax the whole difference
        # at peg = 1, we allow arbitrage
        T = abs((exec_price - args["soft_peg_price"]) * (1 -
                args["arbitrage_coef"]) if args["soft_peg_price"] == 1 else 1)
        volume = abs(volume if volume > 0 else volume * exec_price)
        # we multiply by 2, so we get to 100% tax at a 50% price difference and stick to 100% max
        tax_percentage = 2 * T / args["soft_peg_price"] if 2 * \
            T / args["soft_peg_price"] < 1 else 1
        tax = tax_percentage * abs(volume)
        return round(tax * args["cex_tax_coef"], 5), round(tax * (1-args["cex_tax_coef"]), 5)
    else:
        return 0, 0


def time_peg_increase(
    function: List[float], idx: int, trade_history: List[dict], args: dict
) -> bool:
    # if we are at peg of 1, we dont increase anymore
    if args["soft_peg_price"] >= 1:
        return False
    num_steps = (1 - args["soft_peg_start"]) / args["soft_peg_increase"] + 1
    index_delta = len(function) / num_steps
    last_change = args["last_peg_change"] if "last_peg_change" in args else 0
    if idx >= last_change + index_delta:
        args["soft_peg_price"] += args["soft_peg_increase"]
        args["last_peg_change"] = idx
        return True
    return False


def avg_price_peg_increase(
    function: List[float], idx: int, trade_history: List[dict], args: dict
) -> bool:
    # if we are at peg of 1, we dont increase anymore
    if args["soft_peg_price"] >= 1:
        return False
    # go through history and sum only trades (no arb/buy back trades)
    trade_idx = len(trade_history) - 1
    point_count = 0  # to store trade points we collected
    price_sum = 0
    while trade_idx >= 0:
        trade = trade_history[trade_idx]
        # we want non buy back and non arb trades only
        if trade["buy_back_vol"] == 0 and "arb_trade" not in trade:
            point_count += 1
            price_sum += trade["price"]
            if point_count == args["soft_peg_moving_window"]:
                break
        trade_idx -= 1
        # we dont have enough price points, cannot calculate avg => not increase peg
        if trade_idx < 0:
            return False
    # we dont have enough price points
    if point_count < args["soft_peg_moving_window"]:
        return False
    avg_price = price_sum / args["soft_peg_moving_window"]
    if avg_price >= args["soft_peg_price"]:
        # change the peg
        args["soft_peg_price"] += args["soft_peg_increase"]
        # cap at 1$
        if args["soft_peg_price"] > 1:
            args["soft_peg_price"] = 1
        # record when we last changed the peg, maybe usefull
        args["last_peg_change"] = idx
        return True
    return False


def div_protocol(
    trade: TradeOrder,
    mkt: MarketPair,
    args: dict,
    dt: datetime.datetime,
    is_buy_back: bool = False,
) -> List[dict]:
    """UNI v2 LP strategy with divergence protocol.

    Args:
        row (dict): The trade data.
        mkt (MarketPair): The market pair for which swaps are to be simulated.

    Returns:
        dict: trade_exec_info.

    """
    trade_exec_info = []

    # do the trade, we skip execute_trade, cause we have already TradeOrder
    mid_price = mkt.mid_price
    _, exec_price = constant_product_swap(mkt, trade)
    executed = {
        "trade_date": dt,
        "side": trade.direction,
        "arb_profit": 0,
        "price": exec_price,
        "price_impact": (mid_price - exec_price) / mid_price,
        "peg": args["soft_peg_price"],
        **mkt.describe(),
    }

    # divergence tax if applicable
    if is_buy_back:
        # buy back is in USDT, so convert to USTC
        executed["buy_back_vol"] = trade.order_size / exec_price
    else:
        cex_profit, chain_profit = calc_taxes(
            exec_price,
            trade.order_size if trade.direction == "buy" else -trade.order_size,
            args,
        )
        executed["cex_profit"] = cex_profit
        executed["chain_profit"] = chain_profit
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
