from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from .exec_engine import ConstantProductEngine, calc_arb_trade_pnl
from .market import TradeOrder

default_agg = {
    "price": "mean",
    "mid_price": "last",
    "mkt_price": "last",
    "avg_price": "last",
    "spread": "last",
    "price_impact": "max",
    "mkt_price_ratio": "last",
    "impermanent_loss": "last",
    "current_base": "last",
    "current_quote": "last",
    "cp_invariant": "last",
    "asset_base_pct": "last",
    "volume_base": "sum",
    "volume_quote": "sum",
    "total_volume_base": "last",
    "total_volume_quote": "last",
    "fees_paid_quote": "sum",
    "total_fees_paid_quote": "last",
    "trade_pnl": "last",
    "total_pnl": "last",
    "trade_pnl_pct": "last",
    "fees_pnl_pct": "last",
    "roi": "last",
    "hold_portfolio": "last",
    "current_portfolio": "last",
}


class Strategy(ABC):
    @abstractmethod
    def execute(self, current_row: dict, sim_data: pd.DataFrame) -> List[dict]:
        return []

    @abstractmethod
    def agg(self, sim_results: pd.DataFrame) -> dict:
        return default_agg


class ArbStrategy(Strategy):
    def __init__(self, cp_amm: ConstantProductEngine):
        self.cp_amm = cp_amm

    def execute(self, current_row: dict, sim_data: pd.DataFrame) -> List[dict]:
        trade_exec_info = []
        arb_trade, exec_price = self.cp_amm.calc_arb_trade(current_row["price"])
        if arb_trade:
            arb_trade_pnl = calc_arb_trade_pnl(
                arb_trade, exec_price, current_row["price"], fees=0
            )
            if arb_trade_pnl > 0:
                exec_info = self.cp_amm.execute_trade(current_row, arb_trade)
                exec_info["arb_profit"] = arb_trade_pnl
                exec_info["arb_volume_quote"] = exec_info["volume_quote"]
                trade_exec_info.append(exec_info)
        return trade_exec_info

    def agg(self, sim_results: pd.DataFrame) -> pd.DataFrame:
        if "arb_profit" in sim_results.columns:
            sim_results["retail_volume_quote"] -= sim_results["arb_volume_quote"]
            return {
                "arb_profit": "sum",
                "arb_volume_quote": "sum",
                "retail_volume_quote": "sum",
            }
        return {}


class SimpleUniV2Strategy(Strategy):
    def __init__(self, cp_amm: ConstantProductEngine, arb_enabled: bool = True):
        self.cp_amm = cp_amm
        self.arb_strategy = ArbStrategy(cp_amm) if arb_enabled else None

    def execute(self, current_row: dict, sim_data: pd.DataFrame) -> List[dict]:
        trade_exec_info = []
        if self.arb_strategy:
            trade_exec_info.extend(self.arb_strategy.execute(current_row, sim_data))

        if current_row["quantity"] != 0:
            trade_order = TradeOrder(
                current_row["quantity"],
                self.cp_amm.mkt.swap_fee,
            )
            trade_exec_info.append(self.cp_amm.execute_trade(current_row, trade_order))
        return trade_exec_info

    def agg(self, sim_results: pd.DataFrame) -> pd.DataFrame:
        if self.arb_strategy:
            arb_agg = self.arb_strategy.agg(sim_results)
            return {
                **default_agg,
                **arb_agg,
            }

        return {
            **default_agg,
        }


class DivProtocolParams:
    def __init__(
        self,
        peg_price: float,
        pct_buy_back: float = 1.0,
        exec_buy_back: bool = True,
        borrow_haircut: float = 0.3,
        margin: float = 0.7,
    ):
        self.peg_price = peg_price
        self.pct_buy_back = pct_buy_back
        self.exec_buy_back = exec_buy_back
        self.borrow_haircut = borrow_haircut
        self.margin = margin


# Rewrite the DivProtocolStrategy class with the updated calc_div_tax function
class DivProtocolStrategy:
    def __init__(self, strategy, strat_params):
        assert strat_params.peg_price > 0.0
        self.reserve = 0.0
        self.reserve_base = 0.0
        self.reserve_quote = 0.0
        self.strategy = strategy
        self.params = strat_params
        self.cp_amm = self.strategy.cp_amm

    @property
    def reserve_account(self) -> dict:
        """Ticker for the trading pair represented by this market."""
        return {
            "reserve_base": self.reserve_base,
            "reserve_quote": self.reserve_quote,
            "reserve_base_quote": self.reserve_base_quote,
            "reserve_account": self.reserve,
        }

    def execute(self, current_row, sim_data):
        trade_exec_info = self.strategy.execute(current_row, sim_data)

        mkt_price = current_row["price"]
        for exec_info in trade_exec_info:
            self.update_exec_info_with_div_tax(
                exec_info=exec_info,
                mkt_price=mkt_price,
            )

        if self.params.exec_buy_back:
            if self.reserve > 0 and self.cp_amm.mkt.mid_price < self.params.peg_price:
                buy_back_trade, borrowed_quote = self.calc_buy_back_trade(mkt_price)
                if buy_back_trade:
                    exec_info = self.perform_buy_back(current_row, buy_back_trade, borrowed_quote)
                    trade_exec_info.append(exec_info)

        return trade_exec_info

    def agg(self, sim_results: pd.DataFrame) -> dict:
        strat_agg = self.strategy.agg(sim_results)

        if "div_tax_pct" in sim_results.columns:
            strat_agg.update(
                {
                    **default_agg,
                    "div_exec_price": "mean",
                    "no_div_mid_price": "last",
                    "div_volume_quote": "sum",
                    "div_tax_pct": "mean",
                    "div_tax_quote": "sum",
                    "reserve_base": "last",
                    "reserve_quote": "last",
                    "reserve_base_quote": "last",
                    "reserve_account": "last",
                }
            )

        if "buy_back_volume_quote" in sim_results.columns:
            strat_agg.update(
                {
                    "buy_back_volume_quote": "sum",
                    "borrowed_quote": "sum",
                    "net_borrowed_quote": "sum",
                }
            )

        return strat_agg

    def calc_div_tax(self, mid_price, volume_base, volume_quote):
        peg_price = self.params.peg_price
        div_tax_pct = min(abs(1 - (mid_price / peg_price)), 1.0)
        div_volume_quote = volume_quote * (1 - div_tax_pct)
        return {
            "div_tax_pct": div_tax_pct,
            "div_volume_quote": div_volume_quote,
            "div_tax_quote": div_tax_pct * volume_quote,
            "div_exec_price": div_volume_quote / volume_base,
        }

    def perform_buy_back(self, current_row, buy_back_trade, borrowed_quote):
        mkt_price = current_row["price"]
        no_div_mid_price = self.cp_amm.mkt.mid_price
        exec_info = self.cp_amm.execute_trade(current_row, buy_back_trade)

        cash_quote = buy_back_trade.order_size - borrowed_quote
        net_volume_quote = (mkt_price*exec_info["volume_base"])
        if borrowed_quote > 0:
            net_volume_quote -= borrowed_quote

        self.update_reserve(
            volume_quote=-cash_quote,
            volume_base=net_volume_quote/mkt_price,
            mkt_price=mkt_price,
        )

        exec_info.update(
            {
                "no_div_mid_price": no_div_mid_price,
                "buy_back_volume_quote": exec_info["volume_quote"],
                "borrowed_quote": borrowed_quote,
                "net_borrowed_quote": net_volume_quote,
                **self.reserve_account,
            }
        )

        return exec_info

    def calc_buy_back_trade(self, mkt_price):
        dx, _ = self.cp_amm.mkt.get_delta_reserves(self.params.peg_price)
        if dx <= 0 or self.reserve == 0:
            return None, 0

        borrowed_quote = 0
        order_size = min(dx, self.reserve_quote)
        amount_to_borrow = dx - order_size
        if amount_to_borrow > 0 and self.reserve_base > 0.0:
            borrowed_quote = self.calc_borrow_quote(mkt_price, order_size, amount_to_borrow)
            if borrowed_quote > 0:
                order_size += borrowed_quote

        trade = TradeOrder(order_size, self.cp_amm.mkt.swap_fee) if order_size > 0 else None
        return trade, borrowed_quote


    def calc_borrow_quote(self, mkt_price, cash_quote, amount_to_borrow):
        margin = self.params.margin
        borrow_haircut = self.params.borrow_haircut
        haircut_collat_quote = amount_to_borrow / ((1/margin)-1)
        collat_quote = haircut_collat_quote / (1-borrow_haircut)
        collat_quote = min(collat_quote, self.reserve_base*mkt_price)
        borrowed_quote = collat_quote*(1-borrow_haircut)*((1/margin)-1)
        pnl = self.calc_borrow_trade_pnl(mkt_price, cash_quote, borrowed_quote)
        return borrowed_quote if pnl > 0 else 0

    def calc_borrow_trade_pnl(self, mkt_price, cash_quote, borrowed_quote):
        volume_quote = cash_quote + borrowed_quote
        exec_price, _ = self.cp_amm.get_exec_price(volume_quote)
        volume_base = volume_quote/exec_price
        return (volume_base*mkt_price) - borrowed_quote


    def update_reserve(self, volume_base, volume_quote, mkt_price):
        self.reserve_base += volume_base
        self.reserve_quote += volume_quote
        self.reserve_base_quote = self.reserve_base * mkt_price
        self.reserve = self.reserve_quote + self.reserve_base_quote

    def update_exec_info_with_div_tax(self, exec_info, mkt_price):
        div_exec_info = self.calc_div_tax(
            mid_price=exec_info["mid_price"],
            volume_base=exec_info["volume_base"],
            volume_quote=exec_info["volume_quote"],
        )

        if div_exec_info and div_exec_info["div_tax_pct"] < 0.5:
            self.update_reserve(
                volume_base=0.0,
                volume_quote=div_exec_info["div_tax_quote"],
                mkt_price=mkt_price,
            ),

            exec_info.update(div_exec_info)

        exec_info.update(
            {
                **self.reserve_account,
                "no_div_mid_price": exec_info["mid_price"],
            }
        )
