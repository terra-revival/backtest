from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from .exec_engine import ConstantProductEngine, calc_arb_trade_pnl
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
        arb_trade, exec_price = self.cp_amm.calc_arb_trade(current_row["price"])
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
        if self.arb_strategy:
            trade_exec_info.extend(self.arb_strategy.execute(current_row, sim_data))

        if current_row["quantity"] != 0:
            trade_order = TradeOrder(
                current_row["quantity"],
                self.cp_amm.mkt.swap_fee,
            )
            trade_exec_info.append(self.cp_amm.execute_trade(current_row, trade_order))
        return trade_exec_info

    def agg_results(self, sim_results: pd.DataFrame) -> pd.DataFrame:
        if self.arb_strategy:
            return self.arb_strategy.agg_results(sim_results)
        return sim_results


class DivProtocolParams:
    def __init__(
        self,
        peg_price: float,
        pct_buy_back: float = 1.0,
    ):
        self.peg_price = peg_price
        self.pct_buy_back = pct_buy_back

# Rewrite the DivProtocolStrategy class with the updated calc_div_tax function
class DivProtocolStrategy(Strategy):
    def __init__(
        self, strategy: SimpleUniV2Strategy,
        strat_params: DivProtocolParams,
    ):
        assert strat_params.peg_price > 0.0

        self.strategy = strategy
        self.params = strat_params
        self.reserve_account = 0.0
        self.cp_amm = self.strategy.cp_amm


    def execute(self, current_row: dict, sim_data: pd.DataFrame) -> List[dict]:
        trade_exec_info = self.strategy.execute(current_row, sim_data)
        for exec_info in trade_exec_info:
            div_exec_info = self.calc_div_tax(exec_info)
            if div_exec_info:
                exec_info.update(div_exec_info)
                self.reserve_account += div_exec_info["div_tax_quote"]

            exec_info["reserve_account"] = self.reserve_account
            exec_info["no_div_mid_price"] = exec_info["mid_price"]

        # buy back
        if  self.cp_amm.mkt.mid_price < self.params.peg_price:
            # calculate buy back volume
            no_div_mid_price = self.cp_amm.mkt.mid_price
            dx, _ = self.cp_amm.mkt.get_delta_reserves(self.params.peg_price)
            buy_back_volume_quote = min(self.reserve_account, dx)
            self.reserve_account -= buy_back_volume_quote
            # execute buy back
            trade_order = TradeOrder(buy_back_volume_quote, self.cp_amm.mkt.swap_fee)
            exec_info = self.cp_amm.execute_trade(current_row, trade_order)
            # update exec info
            exec_info["no_div_mid_price"] = no_div_mid_price
            exec_info["reserve_account"] = self.reserve_account
            exec_info["buy_back_volume_quote"] = buy_back_volume_quote
            trade_exec_info.append(exec_info)

        return trade_exec_info

    def agg_results(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.strategy.agg_results(df)
        # if "div_tax_quote" in sim_results.columns:
        #     sim_results["total_div_tax"] = sim_results["div_tax_quote"].cumsum()
        #     sim_results["total_adj_qty_received"] = sim_results[
        #         "adj_qty_received"
        #     ].cumsum()
        return df


    def calc_div_tax(self, exec_info: dict) -> dict:
        """Calculates the divergence protocol tax percentage."""
        # check div conditions
        peg_price = self.params.peg_price
        mid_price = exec_info["mid_price"]
        volume_quote = exec_info["volume_quote"]
        volume_base = exec_info["volume_base"]
        div_tax_pct = min(abs(1 - ( mid_price / peg_price)), 1.0)
        div_volume_quote = volume_quote * (1 - div_tax_pct)
        return {
            "div_tax_pct": div_tax_pct,
            "div_volume_quote": div_volume_quote,
            "div_tax_quote": div_tax_pct*volume_quote,
            "div_exec_price": div_volume_quote / volume_base,
        }
