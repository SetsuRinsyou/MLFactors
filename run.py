"""加载 CSV 数据并计算注册因子。"""

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from dataloader import load_data
from factors_eval import FactorEvalResult, eval as evaluate_factor
from factors.registry import FactorRegistry
from plot import FactorPlotter


class Runner:
    """管理因子的数据加载、计算和结果查看。"""

    def __init__(
        self,
        factor_name: str,
        factor_params: dict[str, Any] | None = None,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
        data_columns: list[str] | None = None,
        forward_periods: tuple[int, ...] = (1, 5, 10, 21),
        n_groups: int = 5,
        ic_method: str = "rank",
        max_lag: int = 20,
        output_dir: str | Path | None = None,
    ) -> None:
        self.factor_name = factor_name
        self.symbols = symbols
        self.start = start
        self.end = end
        self.data_columns = data_columns
        self.forward_periods = forward_periods
        self.n_groups = n_groups
        self.ic_method = ic_method
        self.max_lag = max_lag
        self.output_dir = Path(output_dir or Path("outputs") / factor_name)
        self.data_dir: str | Path | None = None
        self.factor = FactorRegistry.get(factor_name)(**(factor_params or {}))
        self.data = pd.DataFrame()
        self.factor_result = pd.DataFrame()
        self.benchmark = pd.DataFrame()
        self.constituents: dict[str, set[str]] = {}
        self.result = pd.DataFrame()
        self.evaluations: dict[int, FactorEvalResult] = {}
        self.summary = pd.DataFrame()

    def load(self,
             data_dir: str | Path,
             constituents_path: str | Path | None = None,) -> pd.DataFrame:
        self.data_dir = data_dir
        self.data, self.constituents = load_data(
            data_dir=data_dir,
            symbols=self.symbols,
            start=self.start,
            end=self.end,
            constituents_path=constituents_path,
            columns=self.data_columns,
        )
        return self.data

    def load_factor_result(self, factor_dir: str | Path ) -> pd.DataFrame:
        """加载已保存的因子数据。"""
        self.factor_result, _ = load_data(
            data_dir=factor_dir,
            symbols=self.symbols,
            start=self.start,
            end=self.end,
        )
        return self.factor_result

    def load_benchmark(self) -> pd.DataFrame:
        """提取回测期间的 HS300 收盘价。"""
        source_symbol = self.data.index.get_level_values("symbol")[0]
        benchmark_data, _ = load_data(
            data_dir=self.data_dir,
            symbols=[source_symbol],
            start=self.start,
            end=self.end,
            columns=["HS300_close"],
        )
        self.benchmark = benchmark_data.xs(source_symbol, level="symbol").rename(
            columns={"HS300_close": "HS300"}
        )[["HS300"]]
        return self.benchmark

    def calculate(self, save: bool = False) -> pd.DataFrame:
        """计算因子；save=True 时按股票保存完整回测期因子值。"""
        self.result = self.factor.generate_signals(self.data, self.constituents)
        if save:
            factor_dir = self.output_dir / "factor"
            factor_dir.mkdir(parents=True, exist_ok=True)
            symbols = self.data.index.get_level_values("symbol").unique()
            factor_result = self.result.reindex(columns=symbols)
            for symbol in symbols:
                dates = self.data.xs(symbol, level="symbol").index
                factor_data = factor_result[symbol].reindex(dates).rename(self.factor_name)
                factor_data.index = factor_data.index.strftime("%Y-%m-%d")
                factor_data.index.name = "date"
                factor_data.to_csv(
                    factor_dir / f"{symbol}.csv",
                    na_rep="",
                )
        return self.result

    def evaluate(self) -> dict[int, FactorEvalResult]:
        """计算 1、5、10、21 日等指定周期的因子评估结果。"""
        self.evaluations = {
            period: evaluate_factor(
                self.result,
                self.data,
                forward_period=period,
                n_groups=self.n_groups,
                ic_method=self.ic_method,
                max_lag=self.max_lag,
            )
            for period in self.forward_periods
        }
        self.summary = pd.concat(
            [evaluation.summary for evaluation in self.evaluations.values()]
        ).sort_index()
        return self.evaluations

    def save_reports(self) -> Path:
        """保存 CSV、综合评估图和包含表格与图片的 Markdown 报告。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary.to_csv(self.output_dir / "factor_summary.csv")

        image_files = []
        for period, evaluation in self.evaluations.items():
            output_path = self.output_dir / f"{self.factor_name}_{period}d.png"
            benchmark_returns = (
                self.benchmark.shift(-(1 + period))
                / self.benchmark.shift(-1)
                - 1
            )
            benchmark_returns = benchmark_returns.reindex(
                evaluation.layered.group_returns.index
            )
            benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
            FactorPlotter(
                evaluation,
                factor_name=self.factor_name,
                benchmark_cumulative=benchmark_cumulative,
            ).save(output_path)
            image_files.append(output_path)

        report_lines = [
            f"# {self.factor_name} 因子评估报告",
            "",
            "## 运行配置",
            "",
            f"- 数据区间：{self.data.index.get_level_values('date').min().date()} 至 "
            f"{self.data.index.get_level_values('date').max().date()}",
            f"- 股票数量：{self.data.index.get_level_values('symbol').nunique()}",
            f"- 前向收益周期：{', '.join(f'{period} 日' for period in self.forward_periods)}",
            f"- 分层数量：{self.n_groups}",
            f"- IC 方法：{self.ic_method}",
            "",
            "## 多周期评估汇总",
            "",
            self.summary.reset_index().to_markdown(index=False),
            "",
            "## 评估图表",
            "",
        ]
        for period, image_file in zip(self.evaluations, image_files):
            report_lines.extend([
                f"### {period} 日前向收益",
                "",
                f"![{self.factor_name} {period} 日评估图]({image_file.name})",
                "",
            ])

        report_path = self.output_dir / "report.md"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        return report_path

    def latest(self) -> pd.Series:
        """返回最近一个有因子结果的交易日。"""
        latest_result = self.result.dropna(how="all")
        if latest_result.empty:
            return pd.Series(dtype=float, name=self.factor_name)
        result = latest_result.iloc[-1].dropna().sort_values()
        result.name = latest_result.index[-1]
        return result

    def run(self,
            data_dir: str | Path = Path(__file__).resolve().parent / "cache" / "csv",
            factor_dir: str | Path | None = None,
            constituents_path: str | Path | None = None, save_factor: bool = False) -> dict[int, FactorEvalResult]:
        """依次加载数据、计算因子、执行多周期评估并保存结果。"""
        self.load(data_dir=data_dir, constituents_path=constituents_path)
        if factor_dir is not None:
            self.load_factor_result(factor_dir=factor_dir)
            self.data = self.data.join(self.factor_result, how="left")
        self.load_benchmark()
        self.calculate(save=save_factor)
        self.evaluate()
        self.save_reports()
        return self.evaluations


if __name__ == "__main__":
    import factors.risk.beta  # noqa: F401
    import factors.fundamental.cfp_1d  # noqa: F401
    import factors.fundamental.current_asset_growth_12m  # noqa: F401
    import factors.fundamental.current_asset_growth_vol_24m  # noqa: F401
    import factors.fundamental.current_liability_ratio_1d  # noqa: F401
    import factors.fundamental.current_ratio_1d  # noqa: F401
    import factors.fundamental.debt_per_share_1d  # noqa: F401
    import factors.fundamental.debt_to_asset_1d  # noqa: F401
    import factors.fundamental.debt_to_asset_change_12m  # noqa: F401
    import factors.fundamental.dgmarq_12m  # noqa: F401
    import factors.fundamental.dgpoaq_12m  # noqa: F401
    import factors.price.dip_rebound  # noqa: F401
    import factors.fundamental.dividend_yield_1d  # noqa: F401
    import factors.fundamental.droaq_12m  # noqa: F401
    import factors.fundamental.eps_growth_12m  # noqa: F401
    import factors.fundamental.eps_growth_63d  # noqa: F401
    import factors.fundamental.eps_growth_accel_3m  # noqa: F401
    import factors.fundamental.eps_growth_accel_price_adj_3m  # noqa: F401
    import factors.fundamental.eps_growth_accel_vol_adj_3m  # noqa: F401
    import factors.fundamental.eps_growth_price_adj_3m  # noqa: F401
    import factors.fundamental.eps_growth_vol_adj_3m  # noqa: F401
    import factors.fundamental.equity_growth_63d  # noqa: F401
    import factors.fundamental.equity_ratio_change_12m  # noqa: F401
    import factors.fundamental.equity_to_fixed_asset_change_12m  # noqa: F401
    import factors.fundamental.financial_expense_ratio_1d  # noqa: F401
    import factors.fundamental.fixed_ratio_1d  # noqa: F401
    import factors.fundamental.gpoaq_3m  # noqa: F401
    import factors.fundamental.gross_margin_1d  # noqa: F401
    import factors.fundamental.gross_profitability_growth_12m  # noqa: F401
    import factors.fundamental.gross_profitability_trend_qoq_12m  # noqa: F401
    import factors.fundamental.gross_profitability_trend_yoy_48m  # noqa: F401
    import factors.risk.high_low_volatility  # noqa: F401
    import factors.sector.industry_current_asset_growth_stability_24m  # noqa: F401
    import factors.fundamental.inventory_turnover_1d  # noqa: F401
    import factors.fundamental.long_term_debt_ratio_1d  # noqa: F401
    import factors.fundamental.long_term_debt_ratio_change_12m  # noqa: F401
    import factors.price.max_daily_return  # noqa: F401
    import factors.price.multi_horizon_llt_daily  # noqa: F401
    import factors.price.multi_horizon_ma_daily  # noqa: F401
    import factors.fundamental.net_income_cash_ratio_1d  # noqa: F401
    import factors.fundamental.net_income_growth_63d  # noqa: F401
    import factors.fundamental.net_profit_margin_1d  # noqa: F401
    import factors.fundamental.non_current_asset_growth_12m  # noqa: F401
    import factors.fundamental.non_current_asset_growth_vol_24m  # noqa: F401
    import factors.price.open_high_surge_10  # noqa: F401
    import factors.fundamental.operating_expense_ratio_1d  # noqa: F401
    import factors.price.pair_reversal  # noqa: F401
    import factors.fundamental.quick_ratio_1d  # noqa: F401
    import factors.price.reversal  # noqa: F401
    import factors.fundamental.revenue_growth_63d  # noqa: F401
    import factors.fundamental.roe_growth_63d  # noqa: F401
    import factors.risk.r_std  # noqa: F401
    import factors.risk.semi_beta  # noqa: F401
    import factors.price.slp_reversal  # noqa: F401
    import factors.risk.smallcap_growth  # noqa: F401
    import factors.fundamental.sp_1d  # noqa: F401
    import factors.sector.style  # noqa: F401
    import factors.fundamental.total_asset_growth_63d  # noqa: F401
    import factors.fundamental.total_asset_growth_vol_24m  # noqa: F401
    import factors.fundamental.total_asset_turnover_1d  # noqa: F401
    import factors.price.turnover_cv  # noqa: F401
    import factors.risk.vff3_volatility  # noqa: F401
    import factors.price.volume_price_corr  # noqa: F401
    import scheme.MaxICWeight

    factor_configs = {
        "beta_252d": {
            "params": {"lookback": 252, "min_obs": 120, "clip": (-3.0, 3.0)},
            "columns": ["close", "market_cap"],
        },
        "beta_mn_252d": {
            "params": {"window": 252, "min_obs": 120, "min_market_obs": 100},
            "columns": ["close", "market_cap"],
        },
        "beta_mp_252d": {
            "params": {"window": 252, "min_obs": 120, "min_market_obs": 100},
            "columns": ["close", "market_cap"],
        },
        "beta_n_252d": {
            "params": {"window": 252, "min_obs": 120, "min_market_obs": 100},
            "columns": ["close", "market_cap"],
        },
        "beta_p_252d": {
            "params": {"window": 252, "min_obs": 120, "min_market_obs": 100},
            "columns": ["close", "market_cap"],
        },
        "dip_rebound_10d": {
            "params": {"window": 10, "min_periods": 5, "shift_days": 0},
            "columns": ["close", "low"],
        },
        "high_r_std_84d": {
            "params": {"lookback": 84, "min_periods": None},
            "columns": ["close", "high", "low", "volume"],
        },
        "hml_r_std_84d": {
            "params": {"lookback": 84, "min_periods": None},
            "columns": ["close", "high", "low", "volume"],
        },
        "industry_momentum_1d": {
            "params": {"period": 1},
            "columns": ["close", "sector"],
        },
        "max_daily_return_5d": {
            "params": {
                "top_n": 5,
                "min_obs": None,
                "lookback": 21,
                "rebalance_step": 21,
            },
            "columns": ["close"],
        },
        # "multi_horizon_llt_50d": {
        #     "params": {"periods": None, "lookback_days": 50},
        #     "columns": ["close"],
        # },
        # "multi_horizon_ma_50d": {
        #     "params": {"periods": None, "lookback_days": 50},
        #     "columns": ["close"],
        # },
        "open_high_surge_10d": {
            "params": {"window": 10, "min_periods": 5, "shift_days": 0},
            "columns": ["close", "open", "high"],
        },
        "pair_reversal_60d": {
            "params": {"formation_period": 60, "reversal_period": 5},
            "columns": ["close"],
        },
        "r_std_84d": {
            "params": {"lookback": 84, "min_periods": None},
            "columns": ["close", "volume"],
        },
        "reversal_1m": {"params": {}, "columns": ["close"]},
        "reversal_3m": {"params": {}, "columns": ["close"]},
        "reversal_6m": {"params": {}, "columns": ["close"]},
        "size_factor_1d": {"params": {}, "columns": ["close", "market_cap"]},
        "size_squared_1d": {"params": {}, "columns": ["close", "market_cap"]},
        "slp_reversal_20d": {"params": {"window": 20}, "columns": ["close"]},
        "smallcap_growth_1d": {
            "params": {"lower": 0.01, "upper": 0.99, "use_rank": False},
            "columns": ["close", "market_cap"],
        },
        "style_category_momentum_1d": {
            "params": {"period": 1, "n_clusters": 30},
            "columns": ["close", "market_cap", "pe_ratio"],
        },
        "trading_amount_20d": {"params": {}, "columns": ["close", "volume"]},
        "turnover_cv_20d": {
            "params": {"window": 20, "min_periods": 15},
            "columns": ["close", "volume", "market_cap"],
        },
        "vff3_63d": {
            "params": {"lookback": 63, "min_periods": 40, "min_residuals": 10},
            "columns": ["close", "volume", "market_cap", "pb_ratio"],
        },
        "vff3_up_63d": {
            "params": {"lookback": 63, "min_periods": 40, "min_residuals": 10},
            "columns": ["close", "volume", "market_cap", "pb_ratio"],
        },
        "vff3_upd_63d": {
            "params": {"lookback": 63, "min_periods": 40, "min_residuals": 10},
            "columns": ["close", "volume", "market_cap", "pb_ratio"],
        },
        "volume_3m": {"params": {}, "columns": ["close", "volume"]},
        "volume_price_corr_15d": {
            "params": {"window": 10},
            "columns": ["close", "volume", "market_cap"],
        },
        "cfp_1d": {
            "params": {},
            "columns": ["close", "operating_cash_flow", "market_cap", "sector"],
        },
        "current_asset_growth_12m": {
            "params": {},
            "columns": ["close", "current_assets"],
        },
        "current_asset_growth_vol_24m": {
            "params": {},
            "columns": ["close", "current_assets"],
        },
        "current_liability_ratio_1d": {
            "params": {},
            "columns": ["close", "total_liabilities", "current_liabilities", "sector"],
        },
        "current_ratio_1d": {
            "params": {},
            "columns": ["close", "current_assets", "current_liabilities", "sector"],
        },
        "debt_per_share_1d": {
            "params": {},
            "columns": ["close", "total_debt", "shares_basic", "sector"],
        },
        "debt_to_asset_1d": {
            "params": {},
            "columns": ["close", "total_assets", "total_liabilities", "sector"],
        },
        "debt_to_asset_change_12m": {
            "params": {},
            "columns": ["close", "total_liabilities", "total_assets"],
        },
        "dgmarq_12m": {
            "params": {"year_days": 252},
            "columns": ["close", "revenue", "cost_revenue"],
        },
        "dgpoaq_12m": {
            "params": {"year_days": 252},
            "columns": ["close", "total_assets", "revenue", "cost_revenue"],
        },
        "dividend_yield_1d": {
            "params": {},
            "columns": ["close", "dividends_paid", "market_cap", "sector"],
        },
        "droaq_12m": {
            "params": {"year_days": 252},
            "columns": ["close", "roa"],
        },
        "eps_growth_12m": {
            "params": {"year_days": 252},
            "columns": ["close", "eps"],
        },
        "eps_growth_63d": {
            "params": {},
            "columns": ["close", "eps", "sector"],
        },
        "eps_growth_accel_3m": {
            "params": {"year_days": 252, "period_days": 63},
            "columns": ["close", "eps"],
        },
        "eps_growth_accel_price_adj_3m": {
            "params": {"year_days": 252, "period_days": 63},
            "columns": ["close", "eps"],
        },
        "eps_growth_accel_vol_adj_3m": {
            "params": {
                "year_days": 252,
                "period_days": 63,
                "quarter_days": 63,
                "n_quarters": 8,
            },
            "columns": ["close", "eps"],
        },
        "eps_growth_price_adj_3m": {
            "params": {"year_days": 252, "period_days": 63},
            "columns": ["close", "eps"],
        },
        "eps_growth_vol_adj_3m": {
            "params": {"year_days": 252, "quarter_days": 63, "n_quarters": 8},
            "columns": ["close", "eps"],
        },
        "equity_growth_63d": {
            "params": {},
            "columns": ["close", "shareholder_equity", "sector"],
        },
        "equity_ratio_change_12m": {
            "params": {},
            "columns": ["close", "shareholder_equity", "total_assets"],
        },
        "equity_to_fixed_asset_change_12m": {
            "params": {},
            "columns": ["close", "shareholder_equity", "ppe_net"],
        },
        "financial_expense_ratio_1d": {
            "params": {},
            "columns": ["close", "revenue", "interest_expense", "sector"],
        },
        "fixed_ratio_1d": {
            "params": {},
            "columns": ["close", "total_assets", "ppe_net", "sector"],
        },
        "gpoaq_3m": {
            "params": {"quarter_days": 63},
            "columns": ["close", "total_assets", "revenue", "cost_revenue"],
        },
        "gross_margin_1d": {
            "params": {},
            "columns": ["close", "revenue", "gross_profit", "sector"],
        },
        "gross_profitability_growth_12m": {
            "params": {},
            "columns": ["close", "gross_profit", "total_assets"],
        },
        "gross_profitability_trend_qoq_12m": {
            "params": {},
            "columns": ["close", "gross_profit", "total_assets"],
        },
        "gross_profitability_trend_yoy_48m": {
            "params": {},
            "columns": ["close", "gross_profit", "total_assets"],
        },
        "industry_current_asset_growth_stability_24m": {
            "params": {},
            "columns": ["close", "current_assets", "sector"],
        },
        "inventory_turnover_1d": {
            "params": {},
            "columns": ["close", "cost_revenue", "inventory", "sector"],
        },
        "long_term_debt_ratio_1d": {
            "params": {},
            "columns": ["close", "total_assets", "non_current_debt", "sector"],
        },
        "long_term_debt_ratio_change_12m": {
            "params": {},
            "columns": ["close", "non_current_debt", "total_assets"],
        },
        "net_income_cash_ratio_1d": {
            "params": {},
            "columns": ["close", "net_income", "operating_cash_flow", "sector"],
        },
        "net_income_growth_63d": {
            "params": {},
            "columns": ["close", "net_income", "sector"],
        },
        "net_profit_margin_1d": {
            "params": {},
            "columns": ["close", "net_income", "revenue", "sector"],
        },
        "non_current_asset_growth_12m": {
            "params": {},
            "columns": ["close", "non_current_assets"],
        },
        "non_current_asset_growth_vol_24m": {
            "params": {},
            "columns": ["close", "non_current_assets"],
        },
        "operating_expense_ratio_1d": {
            "params": {},
            "columns": ["close", "operating_expense", "revenue", "sector"],
        },
        "quick_ratio_1d": {
            "params": {},
            "columns": [
                "close",
                "current_assets",
                "current_liabilities",
                "inventory",
                "sector",
            ],
        },
        "revenue_growth_63d": {
            "params": {},
            "columns": ["close", "revenue", "sector"],
        },
        "roe_growth_63d": {
            "params": {},
            "columns": ["close", "roe", "sector"],
        },
        "sp_1d": {
            "params": {},
            "columns": ["close", "market_cap", "revenue", "sector"],
        },
        "total_asset_growth_63d": {
            "params": {},
            "columns": ["close", "total_assets", "sector"],
        },
        "total_asset_growth_vol_24m": {
            "params": {},
            "columns": ["close", "total_assets"],
        },
        "total_asset_turnover_1d": {
            "params": {},
            "columns": ["close", "total_assets", "revenue", "sector"],
        },
        # "max_ic_weight_10d": {"params": {"ic_window": 10},
        #                       "columns": ["close"]}
    }

    for factor_name, config in factor_configs.items():
        runner = Runner(
            factor_name=factor_name,
            factor_params=config["params"],
            symbols=None,
            start="2017-01-01",
            n_groups=5,
            output_dir=f"outputs/hs300/{factor_name}",
            data_columns=config["columns"],
        )
        runner.run(save_factor=True,
                   data_dir="cache/hs300_csv")
        print(f"{factor_name} 已保存到: {runner.output_dir.resolve()}")

        runner = Runner(
            factor_name=factor_name,
            factor_params=config["params"],
            symbols=None,
            start="2017-01-01",
            n_groups=5,
            output_dir=f"outputs/zz500/{factor_name}",
            data_columns=config["columns"],
        )
        runner.run(save_factor=True,
                   data_dir="cache/zz500_csv")
        print(f"{factor_name} 已保存到: {runner.output_dir.resolve()}")

        runner = Runner(
            factor_name=factor_name,
            factor_params=config["params"],
            symbols=None,
            start="2017-01-01",
            n_groups=5,
            output_dir=f"outputs/zz1000/{factor_name}",
            data_columns=config["columns"],
        )
        runner.run(save_factor=True,
                   data_dir="cache/zz1000_csv")
        print(f"{factor_name} 已保存到: {runner.output_dir.resolve()}")

    # 组合因子回测
    # scheme_configs = {
    #     "max_ic_weight_10d": {"params": {"ic_window": 10},
    #                           "columns": ["close"]}
    # }

    # for scheme_name, config in scheme_configs.items():
    #     runner = Runner(
    #         factor_name=scheme_name,
    #         factor_params=config["params"],
    #         symbols=None,
    #         start="2017-01-01",
    #         n_groups=5,
    #         output_dir=f"outputs/{scheme_name}",
    #         data_columns=config["columns"],
    #     )
    #     runner.run(save_factor=True,
    #                data_dir="cache/hs300_csv")
    #     print(f"{scheme_name} 已保存到: {runner.output_dir.resolve()}")