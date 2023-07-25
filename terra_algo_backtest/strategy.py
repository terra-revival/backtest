from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from .exec_engine import ConstantProductEngine, calc_arb_trade, calc_arb_trade_pnl
from .market import TradeOrder


class Strategy(ABC):
    @abstractmethod
    def execute(self, current_row: dict, sim_data: pd.DataFrame) -> List[dict]:
        pass


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
