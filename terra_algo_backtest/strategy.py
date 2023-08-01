from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from .exec_engine import ConstantProductEngine, calc_arb_trade, calc_arb_trade_pnl
from .market import TradeOrder


class Strategy(ABC):
    @abstractmethod
    def execute(self, current_row: dict, sim_data: pd.DataFrame) -> List[dict]:
        return []

    @abstractmethod
    def agg_results(self, sim_results: pd.DataFrame) -> pd.DataFrame:
        return sim_results


class ArbStrategy(Strategy):
    def __init__(self, cp_amm: ConstantProductEngine):
        self.cp_amm = cp_amm

    def execute(self, current_row: dict, sim_data: pd.DataFrame) -> List[dict]:
        trade_exec_info = []
        self.cp_amm.update_mkt_price(current_row["price"])
        arb_trade, exec_price = calc_arb_trade(self.cp_amm)
        if arb_trade:
            arb_trade_pnl = calc_arb_trade_pnl(
                arb_trade, exec_price, current_row["price"], fees=0
            )
            if arb_trade_pnl > 0:
                exec_info = self.cp_amm.execute_trade(current_row, arb_trade)
                exec_info["arb_profit"] = arb_trade_pnl
                trade_exec_info.append(exec_info)
        return trade_exec_info

    def agg_results(self, sim_results: pd.DataFrame) -> pd.DataFrame:
        if "arb_profit" in sim_results.columns:
            sim_results["total_arb_profit"] = sim_results["arb_profit"].cumsum()
        return sim_results


class SimpleUniV2Strategy(Strategy):
    def __init__(self, cp_amm: ConstantProductEngine, arb_enabled: bool = True):
        self.cp_amm = cp_amm
        self.arb_strategy = ArbStrategy(cp_amm) if arb_enabled else None

    def execute(self, current_row: dict, sim_data: pd.DataFrame) -> List[dict]:
        trade_exec_info = []
        self.cp_amm.update_mkt_price(current_row["price"])

        if self.arb_strategy:
            trade_exec_info.extend(self.arb_strategy.execute(current_row, sim_data))

        if current_row["quantity"] != 0:
            trade_order = TradeOrder(
                self.cp_amm.mkt.ticker,
                current_row["quantity"],
                self.cp_amm.mkt.swap_fee,
            )
            trade_exec_info.append(self.cp_amm.execute_trade(current_row, trade_order))
        return trade_exec_info

    def agg_results(self, sim_results: pd.DataFrame) -> pd.DataFrame:
        if self.arb_strategy:
            return self.arb_strategy.agg_results(sim_results)
        return sim_results


# Rewrite the DivProtocolStrategy class with the updated calc_div_tax function
class DivProtocolStrategy(Strategy):
    def __init__(self, strategy: SimpleUniV2Strategy, peg_price: float):
        assert peg_price > 0.0
        self.strategy = strategy
        self.peg_price = peg_price
        self.cp_amm = self.strategy.cp_amm

    def execute(self, current_row: dict, sim_data: pd.DataFrame) -> List[dict]:
        trade_exec_info = self.strategy.execute(current_row, sim_data)
        for exec_info in trade_exec_info:
            div_tax_pct = self.calc_div_tax(exec_info)
            exec_info["buy_back_mid_price"] = exec_info["mid_price"]

            if div_tax_pct > 0.0:
                # div tax
                exec_info["div_tax_pct"] = div_tax_pct
                exec_info["div_tax_quote"] = (
                    exec_info["qty_received"] * div_tax_pct / exec_info["price"]
                )
                exec_info["adj_qty_received"] = exec_info["qty_received"] * (
                    1 - div_tax_pct
                )

                # buy back
                dx, _ = self.cp_amm.mkt.get_delta_reserves(self.peg_price)
                buy_back_qty = min(dx, exec_info["div_tax_quote"])
                exec_price, mid_price = self.cp_amm.get_exec_price("buy", buy_back_qty)

                exec_info["buy_back_price"] = exec_price
                exec_info["buy_back_mid_price"] = mid_price
                exec_info["buy_back_volume_quote"] = buy_back_qty
                exec_info["div_tax_quote_unused"] = (
                    exec_info["div_tax_quote"] - buy_back_qty
                )

        return trade_exec_info

    def agg_results(self, sim_results: pd.DataFrame) -> pd.DataFrame:
        sim_results = self.strategy.agg_results(sim_results)
        if (
            "div_tax" in sim_results.columns
            and "adj_qty_received" in sim_results.columns
        ):
            sim_results["total_div_tax"] = sim_results["div_tax"].cumsum()
            sim_results["total_adj_qty_received"] = sim_results[
                "adj_qty_received"
            ].cumsum()
        return sim_results

    def calc_div_tax(self, exec_info: dict) -> float:
        """Calculates the divergence protocol for a trade and returns tax in USDT.

        Args:
            exec_info (dict): execution info of the trade

        Returns:
            float: tax as a percentage of the quantity received

        """
        # if exec_info["price"] == 0.0:
        #     return 1.0

        if exec_info["side"] == "sell" and exec_info["mid_price"] < self.peg_price:
            return (self.peg_price - exec_info["mid_price"]) / self.peg_price

        # if exec_info["side"] == "buy" and exec_info["price"] > self.peg_price:
        #     return min((exec_info["price"] - self.peg_price) / exec_info["price"], 1.0)

        return 0.0  # Scenario 3: Price is at Peg of $1USD (X=Y)
