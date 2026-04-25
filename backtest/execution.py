"""撮合引擎基类与整数股数示例实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from data.schema import Col


@dataclass
class SimulationResult:
    """撮合引擎输出的回测结果容器。"""

    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class BaseExecutionEngine(ABC):
    """撮合引擎基类。"""

    def __init__(self, commission: float = 0.001, slippage: float = 0.001) -> None:
        self.commission = commission
        self.slippage = slippage

    @abstractmethod
    def run_simulation(
        self,
        target_weights: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> SimulationResult:
        """执行回测撮合。"""
        ...


class SimpleBacktestExecutionEngine(BaseExecutionEngine):
    """最简单的整数股数回测执行器。

    - T 日目标仓位在 T+1 日收盘执行
    - 显式维护现金账户
    - 每次调仓按整数股数交易
    - 成本按成交额比例扣除
    """

    def __init__(
        self,
        commission: float = 0.001,
        slippage: float = 0.001,
        price_col: str | None = None,
        initial_capital: float = 1_000_000.0,
    ) -> None:
        super().__init__(commission=commission, slippage=slippage)
        if initial_capital <= 0:
            raise ValueError("initial_capital 必须为正数")
        self.price_col = price_col
        self.initial_capital = float(initial_capital)

    def _resolve_price_col(self, market_data: pd.DataFrame) -> str:
        if self.price_col is not None:
            if self.price_col not in market_data.columns:
                raise KeyError(f"market_data 中不存在价格列: {self.price_col}")
            return self.price_col
        if Col.ADJ_CLOSE in market_data.columns:
            return Col.ADJ_CLOSE
        if Col.CLOSE in market_data.columns:
            return Col.CLOSE
        raise KeyError("market_data 至少需要包含 adj_close 或 close 列")

    def run_simulation(
        self,
        target_weights: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> SimulationResult:
        raw_prices = (
            market_data[self._resolve_price_col(market_data)]
            .unstack(Col.SYMBOL)
            .sort_index()
        )
        prices = raw_prices.ffill()
        desired = (
            target_weights.reindex(index=prices.index, columns=prices.columns)
            .ffill()
            .fillna(0.0)
        )

        # T 日目标仓位在 T+1 日收盘执行。
        desired = desired.shift(1).fillna(0.0)

        symbols = list(prices.columns)
        shares = pd.Series(0, index=symbols, dtype=int)
        cash = self.initial_capital
        prev_equity = self.initial_capital

        nav_curve: list[float] = []
        returns: list[float] = []
        portfolio_values: list[float] = []
        cash_values: list[float] = []
        share_records: list[pd.Series] = []
        weight_records: list[pd.Series] = []
        trade_rows: list[dict[str, float | pd.Timestamp]] = []
        trade_detail_rows: list[dict[str, float | int | str | pd.Timestamp]] = []

        total_commission = 0.0
        total_slippage = 0.0
        total_turnover = 0.0

        for date in prices.index:
            current_prices = prices.loc[date]
            equity_before = float(cash + (shares * current_prices).sum())
            target_weight_row = desired.loc[date].clip(lower=0.0).fillna(0.0)

            target_shares = pd.Series(0, index=symbols, dtype=int)
            for symbol in symbols:
                price = current_prices[symbol]
                weight = float(target_weight_row[symbol])
                if weight <= 0 or not np.isfinite(price) or price <= 0:
                    continue
                budget = equity_before * weight
                target_shares[symbol] = int(
                    np.floor(budget / (price * (1.0 + self.commission + self.slippage)))
                )

            trade_notional = 0.0
            trade_commission = 0.0
            trade_slippage = 0.0
            buy_shares = 0
            sell_shares = 0

            sell_delta = target_shares - shares
            for symbol in sell_delta[sell_delta < 0].index:
                qty = int(-sell_delta[symbol])
                price = float(current_prices[symbol])
                notional = qty * price
                commission_cost = notional * self.commission
                slippage_cost = notional * self.slippage
                cash += notional - commission_cost - slippage_cost
                shares[symbol] -= qty
                portfolio_value_after_trade = float(cash + (shares * current_prices).sum())

                trade_notional += notional
                trade_commission += commission_cost
                trade_slippage += slippage_cost
                sell_shares += qty
                trade_detail_rows.append(
                    {
                        Col.DATE: date,
                        Col.SYMBOL: str(symbol),
                        "action": "卖出",
                        "shares": qty,
                        "price": price,
                        "notional": notional,
                        "commission_cost": commission_cost,
                        "slippage_cost": slippage_cost,
                        "total_cost": commission_cost + slippage_cost,
                        "cash_after": float(cash),
                        "position_after": int(shares[symbol]),
                        "portfolio_value_after": portfolio_value_after_trade,
                    }
                )

            buy_candidates = target_weight_row[target_weight_row > 0].sort_values(ascending=False).index
            for symbol in buy_candidates:
                qty_needed = int(target_shares[symbol] - shares[symbol])
                if qty_needed <= 0:
                    continue
                price = float(current_prices[symbol])
                unit_cost = price * (1.0 + self.commission + self.slippage)
                affordable = int(cash // unit_cost) if unit_cost > 0 else 0
                qty = min(qty_needed, affordable)
                if qty <= 0:
                    continue

                notional = qty * price
                commission_cost = notional * self.commission
                slippage_cost = notional * self.slippage
                cash -= notional + commission_cost + slippage_cost
                shares[symbol] += qty
                portfolio_value_after_trade = float(cash + (shares * current_prices).sum())

                trade_notional += notional
                trade_commission += commission_cost
                trade_slippage += slippage_cost
                buy_shares += qty
                trade_detail_rows.append(
                    {
                        Col.DATE: date,
                        Col.SYMBOL: str(symbol),
                        "action": "买入",
                        "shares": qty,
                        "price": price,
                        "notional": notional,
                        "commission_cost": commission_cost,
                        "slippage_cost": slippage_cost,
                        "total_cost": commission_cost + slippage_cost,
                        "cash_after": float(cash),
                        "position_after": int(shares[symbol]),
                        "portfolio_value_after": portfolio_value_after_trade,
                    }
                )

            equity_after = float(cash + (shares * current_prices).sum())
            nav = equity_after / self.initial_capital
            ret = (equity_after / prev_equity - 1.0) if prev_equity > 0 else 0.0
            prev_equity = equity_after

            position_weights = (
                (shares * current_prices) / equity_after
                if equity_after > 0 else pd.Series(0.0, index=symbols)
            )
            turnover = trade_notional / equity_before if equity_before > 0 else 0.0

            nav_curve.append(nav)
            returns.append(ret)
            portfolio_values.append(equity_after)
            cash_values.append(float(cash))
            share_records.append(shares.copy())
            weight_records.append(position_weights.fillna(0.0))
            trade_rows.append(
                {
                    Col.DATE: date,
                    "turnover": turnover,
                    "trade_notional": trade_notional,
                    "commission_cost": trade_commission,
                    "slippage_cost": trade_slippage,
                    "total_cost": trade_commission + trade_slippage,
                    "buy_shares": float(buy_shares),
                    "sell_shares": float(sell_shares),
                    "cash": float(cash),
                    "portfolio_value": equity_after,
                    "nav": nav,
                    "net_return": ret,
                }
            )

            total_commission += trade_commission
            total_slippage += trade_slippage
            total_turnover += turnover

        index = prices.index
        positions = pd.DataFrame(share_records, index=index, columns=symbols).astype(int)
        position_weights = pd.DataFrame(weight_records, index=index, columns=symbols).fillna(0.0)
        equity_curve = pd.Series(nav_curve, index=index, name="equity")
        returns_series = pd.Series(returns, index=index, name="strategy_returns")
        trades = pd.DataFrame(trade_rows).set_index(Col.DATE)
        trades.index = pd.to_datetime(trades.index)
        trade_details_columns = [
            Col.DATE,
            Col.SYMBOL,
            "action",
            "shares",
            "price",
            "notional",
            "commission_cost",
            "slippage_cost",
            "total_cost",
            "cash_after",
            "position_after",
            "portfolio_value_after",
        ]
        trade_details = pd.DataFrame(trade_detail_rows, columns=trade_details_columns)
        if not trade_details.empty:
            trade_details[Col.DATE] = pd.to_datetime(trade_details[Col.DATE])

        return SimulationResult(
            equity_curve=equity_curve,
            returns=returns_series,
            positions=positions,
            trades=trades,
            meta={
                "avg_turnover": float(trades["turnover"].mean()) if len(trades) else 0.0,
                "total_turnover": float(total_turnover),
                "total_commission": float(total_commission),
                "total_slippage": float(total_slippage),
                "commission_rate": float(self.commission),
                "slippage_rate": float(self.slippage),
                "initial_capital": float(self.initial_capital),
                "portfolio_value": pd.Series(portfolio_values, index=index, name="portfolio_value"),
                "cash_curve": pd.Series(cash_values, index=index, name="cash"),
                "position_weights": position_weights,
                "trade_details": trade_details,
            },
        )
