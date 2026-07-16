"""因子 IC、换手率和分层收益等基础评估指标。"""

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)


TIMING_WIN_HORIZON_WEIGHTS = {
    5: 0.6,
    10: 0.3,
    21: 0.1,
}


TAIL_FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.25)
DAILY_METRIC_COLUMNS = (
    "IC",
    "TimingIC",
    "top_5%_return",
    "bottom_5%_return",
    "top_10%_return",
    "bottom_10%_return",
    "top_15%_return",
    "bottom_15%_return",
    "top_20%_return",
    "bottom_20%_return",
    "top_25%_return",
    "bottom_25%_return",
)


@dataclass
class LayeredResult:
    """保存分层回测产生的收益和风险指标。

    Attributes
    ----------
    group_returns : pd.DataFrame
        每个调仓日期、每个因子分组的等权平均收益，索引为日期，列为组号。
    top_cumulative_returns, bottom_cumulative_returns : pd.Series
        最高/最低因子分组根据逐期收益复利计算的累计收益。
    top_annual_return, bottom_annual_return : float
        最高/最低因子分组按照实际有效期数和年化频率计算的年化收益。
    top_sharpe_ratio, bottom_sharpe_ratio : float
        最高/最低因子分组不扣除无风险利率的年化夏普比率。
    top_max_drawdown, bottom_max_drawdown : float
        最高/最低因子分组作为多头组合时的最大回撤。
    top_bottom_win_rate : float
        最高因子组收益减最低因子组收益大于 0 的期数占比。
    n_groups : int
        分层数量。
    """

    group_returns: pd.DataFrame
    top_cumulative_returns: pd.Series
    bottom_cumulative_returns: pd.Series
    top_annual_return: float = 0.0
    bottom_annual_return: float = 0.0
    top_sharpe_ratio: float = 0.0
    bottom_sharpe_ratio: float = 0.0
    top_max_drawdown: float = 0.0
    bottom_max_drawdown: float = 0.0
    win_rate: float = 0.0
    n_groups: int = 5


@dataclass
class FactorEvalResult:
    """保存单个因子的完整评估结果。

    Attributes
    ----------
    summary : pd.DataFrame
        单行核心指标汇总表。
    full_ic_series : pd.Series
        每个交易日截面的完整 IC 时间序列。
    daily_metrics : pd.DataFrame
        每个交易日的 IC、TimingIC 和顶部/底部不同选股比例的平均未来收益。
    sampled_ic_series : pd.Series
        按前向收益周期抽样、用于绘图和显著性检验的 IC 时间序列。
    turnover : pd.Series
        最高因子分组组合的单边换手率。
    layered : LayeredResult
        分层收益和风险指标。
    """

    summary: pd.DataFrame
    full_ic_series: pd.Series
    daily_metrics: pd.DataFrame
    sampled_ic_series: pd.Series
    turnover: pd.Series
    layered: LayeredResult


def calc_ic(
    factor: pd.Series,
    returns: pd.Series,
    method: str = "rank",
) -> float:
    """计算一个日期截面上的因子值与未来收益相关系数。

    函数先按索引对齐因子值和收益，再删除任一侧为 NaN 的样本。
    有效股票少于 3 只时无法形成可靠截面，返回 ``np.nan``。

    Parameters
    ----------
    factor : pd.Series
        同一个日期截面上各股票的因子值，索引通常为 symbol。
    returns : pd.Series
        与 ``factor`` 对应的未来收益。
    method : str, default "rank"
        ``"rank"`` 使用 Spearman 秩相关；其他值使用 Pearson 线性相关。

    Returns
    -------
    float
        截面 IC，理论范围为 [-1, 1]；有效样本不足时为 NaN。
    """
    aligned = pd.DataFrame({"factor": factor, "returns": returns}).dropna()
    if len(aligned) < 3:
        return np.nan
    if method == "rank":
        return float(stats.spearmanr(aligned["factor"], aligned["returns"])[0])
    return float(stats.pearsonr(aligned["factor"], aligned["returns"])[0])


def calc_ic_series(
    factor: pd.DataFrame | pd.Series,
    returns: pd.DataFrame | pd.Series,
    method: str = "rank",
) -> pd.Series:
    """对每个日期截面计算 IC，形成按日期排列的 IC 时间序列。

    输入应使用 ``(date, symbol)`` MultiIndex。若传入 DataFrame，函数只使用
    第一列。因子和收益先按完整 MultiIndex 对齐并删除缺失值，再按第一层
    date 分组调用 :func:`calc_ic`。没有任何共同样本时返回空 Series。

    Parameters
    ----------
    factor : pd.Series or pd.DataFrame
        多期因子值，索引为 ``(date, symbol)``。
    returns : pd.Series or pd.DataFrame
        与因子对应的多期未来收益，索引结构相同。
    method : str, default "rank"
        IC 计算方式，``"rank"`` 为 Spearman，否则为 Pearson。

    Returns
    -------
    pd.Series
        名称为 ``IC``、索引为日期的升序时间序列。
    """
    if isinstance(factor, pd.DataFrame):
        factor = factor.iloc[:, 0]
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]

    combined = pd.DataFrame({"factor": factor, "returns": returns}).dropna()
    if combined.empty:
        return pd.Series(dtype=float, name="IC", index=pd.DatetimeIndex([], name="date"))

    def cross_sectional_ic(cross_section: pd.DataFrame) -> float:
        """计算 groupby 传入的单日截面 IC。

        ``cross_section`` 包含已经对齐并删除缺失值的 factor 和 returns
        两列，返回当前日期的相关系数。
        """
        return calc_ic(cross_section["factor"], cross_section["returns"], method)

    result = combined.groupby(level=0).apply(cross_sectional_ic)
    return pd.Series(result, name="IC").sort_index()


def calc_tail_group_returns(
    factor: pd.DataFrame | pd.Series,
    returns: pd.DataFrame | pd.Series,
    fractions: tuple[float, ...] = TAIL_FRACTIONS,
) -> pd.DataFrame:
    """计算每日因子顶部/底部指定比例股票的等权平均未来收益。

    对同一交易日，先将因子值和未来收益对齐并去除缺失值，再按因子值排序。
    每个比例 ``p`` 选取 ``ceil(p * n)`` 只股票；相同因子值使用稳定排序，
    从而保证每个截面始终选出固定数量的顶部和底部股票。

    Parameters
    ----------
    factor, returns
        使用 ``(date, symbol)`` MultiIndex 的因子值和未来收益。
    fractions
        顶部及底部选股比例，例如 ``0.05`` 表示 5%。

    Returns
    -------
    pd.DataFrame
        索引为 date，列按 ``top_<比例>_return``、
        ``bottom_<比例>_return`` 成对排列。每个值均为当日形成组合的
        等权平均未来收益，而非累计收益。
    """
    if isinstance(factor, pd.DataFrame):
        factor = factor.iloc[:, 0]
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    if not fractions or any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError("fractions 必须为 (0, 1] 内的非空比例")

    columns = [
        column
        for fraction in fractions
        for column in (
            f"top_{fraction:.0%}_return",
            f"bottom_{fraction:.0%}_return",
        )
    ]
    combined = pd.DataFrame({"factor": factor, "returns": returns}).dropna()
    if combined.empty:
        return pd.DataFrame(
            columns=columns,
            index=pd.DatetimeIndex([], name="date"),
            dtype=float,
        )

    records: list[dict[str, float | pd.Timestamp]] = []
    for current_date, cross_section in combined.groupby(level=0, sort=True):
        values = cross_section.droplevel(0).sort_values(
            "factor",
            kind="mergesort",
        )
        count = len(values)
        record: dict[str, float | pd.Timestamp] = {"date": current_date}
        for fraction in fractions:
            selected_count = max(1, int(np.ceil(count * fraction)))
            label = f"{fraction:.0%}"
            record[f"top_{label}_return"] = float(
                values.iloc[-selected_count:]["returns"].mean()
            )
            record[f"bottom_{label}_return"] = float(
                values.iloc[:selected_count]["returns"].mean()
            )
        records.append(record)

    return (
        pd.DataFrame.from_records(records)
        .set_index("date")
        .reindex(columns=columns)
        .sort_index()
    )


def calc_offset_ic_stats(
    ic_series: pd.Series,
    period: int = 1,
) -> dict[str, float]:
    """按收益周期拆分 offset 后汇总 IC 均值、标准差和 ICIR。"""
    if period <= 0:
        raise ValueError("period 必须为正整数")
    values = ic_series.sort_index()
    means = []
    stds = []
    icirs = []
    for offset in range(period):
        offset_values = values.iloc[offset::period].dropna()
        if offset_values.empty:
            continue
        means.append(float(offset_values.mean()))
        if len(offset_values) >= 2:
            std = offset_values.std()
            if std > 0:
                stds.append(float(std))
                icirs.append(float(offset_values.mean() / std))
            elif std == 0:
                stds.append(0.0)
    return {
        "IC_mean": float(np.mean(means)) if means else np.nan,
        "IC_std": float(np.mean(stds)) if stds else np.nan,
        "ICIR": float(np.mean(icirs)) if icirs else np.nan,
    }


def calc_ic_half_life(
    ic_series: pd.Series,
    ic_mean: float,
    smoothing_window: int = 5,
) -> float:
    """计算日频 IC 平滑序列的平均半衰期（交易日）。

    先对每日 IC 计算中心化滑动平均
    ``x_t = mean(IC[t-2], ..., IC[t+2])``。为使负向有效因子也能按同一
    "有效性衰减"口径衡量，若全时段 ``IC_mean < 0``，会先将 IC 乘以 -1。
    对每一个有效 ``x_t``，半衰期是第一个满足
    ``x_(t+m) <= x_t / 2`` 且 ``x_(t+m+1) <= x_t / 2`` 的 ``m``；
    已经不具正向有效性的 ``x_t <= 0`` 记为 0。样本末端尚未找到连续两日
    衰减点的观测记为 ``NaN``，不参与平均值。

    Parameters
    ----------
    ic_series
        用指定前向收益周期计算得到的日频 IC 序列。
    ic_mean
        当前回测窗口的 IC 均值，用于确定因子有效方向。
    smoothing_window
        中心化滑动窗口长度；当前口径固定为 5。

    Returns
    -------
    float
        所有可定义局部半衰期的平均交易日数；没有可定义观测时返回 ``NaN``。
    """
    if smoothing_window != 5:
        raise ValueError("当前 IC 半衰期口径固定使用 5 日中心化滑动窗口")
    if not np.isfinite(ic_mean) or ic_mean == 0:
        return np.nan

    values = (
        ic_series.sort_index()
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )
    if values.empty:
        return np.nan

    oriented_ic = values * np.sign(ic_mean)
    smoothed = oriented_ic.rolling(
        window=smoothing_window,
        center=True,
        min_periods=smoothing_window,
    ).mean()
    x_values = smoothed.to_numpy(dtype=float)
    half_lives: list[float] = []

    # 最后两个有效 x_t 不可能拥有一对连续的未来 x，因此不作为起点。
    for position in range(len(x_values) - 2):
        baseline = x_values[position]
        if not np.isfinite(baseline):
            continue
        if baseline <= 0:
            half_lives.append(0.0)
            continue

        threshold = baseline / 2.0
        for future_position in range(position + 1, len(x_values) - 1):
            first = x_values[future_position]
            second = x_values[future_position + 1]
            if (
                np.isfinite(first)
                and np.isfinite(second)
                and first <= threshold
                and second <= threshold
            ):
                half_lives.append(float(future_position - position))
                break

    return float(np.mean(half_lives)) if half_lives else np.nan


def calc_ic_positive_rate(ic_series: pd.Series) -> float:
    """计算有效日频 IC 中严格大于零的交易日占比。"""
    values = (
        ic_series.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )
    return float((values > 0).mean()) if not values.empty else np.nan


def calc_quintile_monotonicity(
    group_returns: pd.DataFrame,
    ic_mean: float,
    n_groups: int = 5,
) -> float:
    """以 IC 方向为基准计算五分组收益单调性。

    先计算每个因子分组的平均未来收益 ``r_1, ..., r_5``，再令相邻差分
    ``d_i = r_(i+1) - r_i``。返回

    ``sign(IC_mean) * sum(d_i) / sum(abs(d_i))``。

    该指标位于 ``[-1, 1]``：``+1`` 表示五组收益完全沿 IC 方向单调，
    ``-1`` 表示完全反向单调。例如 IC_mean 为正但底部组平均收益高于顶部
    组时，指标为负。IC_mean 为零、分组不完整或所有组收益完全相等时返回
    ``NaN``。
    """
    if not np.isfinite(ic_mean) or ic_mean == 0:
        return np.nan
    expected_groups = list(range(1, n_groups + 1))
    if group_returns.empty or not set(expected_groups).issubset(group_returns.columns):
        return np.nan

    means = group_returns[expected_groups].mean().dropna()
    if len(means) != n_groups:
        return np.nan
    differences = means.diff().dropna()
    absolute_change = float(differences.abs().sum())
    if absolute_change == 0:
        return np.nan
    return float(np.sign(ic_mean) * differences.sum() / absolute_change)


def calc_t_stat(
    ic_series: pd.Series,
    period: int = 1,
    maxlags: int | None = None,
) -> tuple[float, float]:
    """使用 Newey-West / HAC 标准误检验 IC 均值是否显著偏离零。

    Parameters
    ----------
    ic_series : pd.Series
        按时间排列的 IC 序列。

    period : int, default=1
        前瞻收益窗口覆盖的 IC 观测期数。

        例如：
        - 每日计算 IC，使用未来 1 日收益：period=1
        - 每日计算 IC，使用未来 5 日收益：period=5
        - 每日计算 IC，使用未来 20 日收益：period=20

    maxlags : int | None, default=None
        HAC 最大滞后阶数。

        默认使用 period - 1，对收益窗口重合导致的序列相关
        进行修正。

        可以显式指定更大的值，以处理 IC 本身额外存在的
        时间序列相关性。

    Returns
    -------
    tuple[float, float]
        HAC 修正后的 ``(t_stat, p_value)``。
    """
    if period <= 0:
        raise ValueError("period 必须为正整数")

    values = (
        ic_series
        .sort_index()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )

    n_obs = len(values)

    if n_obs < 3:
        return np.nan, np.nan

    # h 期收益最多产生 h - 1 阶机械重叠
    if maxlags is None:
        maxlags = period - 1

    if maxlags < 0:
        raise ValueError("maxlags 不能小于 0")

    # HAC 滞后阶数不能超过样本数量减 1
    maxlags = min(maxlags, n_obs - 1)

    # IC_t = alpha + epsilon_t
    # alpha 即 IC 样本均值
    X = np.ones((n_obs, 1), dtype=float)

    result = sm.OLS(
        endog=values.to_numpy(),
        exog=X,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": maxlags,
            "use_correction": True,
        },
        use_t=True,
    )

    t_stat = result.tvalues[0]
    p_value = result.pvalues[0]

    return float(t_stat), float(p_value)

def calc_turnover(
    factor: pd.DataFrame | pd.Series,
    quantiles: int = 5,
) -> pd.Series:
    """计算最高因子分位数组合相邻调仓期之间的单边换手率。

    每个日期先对有效因子值使用 ``rank(method="first")``，再通过 qcut 分成
    ``quantiles`` 组。最高组内股票等权，其余股票权重为零。单边换手率为
    ``sum(abs(weight_t - weight_t-1)) / 2``。截面股票数不足分组数时，该期
    所有权重设为零。

    Parameters
    ----------
    factor : pd.Series or pd.DataFrame
        ``(date, symbol)`` MultiIndex 因子值；DataFrame 只使用第一列。
    quantiles : int, default 5
        截面分组数量，5 表示持有最高 20% 的股票。

    Returns
    -------
    pd.Series
        索引为日期、名称为 ``turnover`` 的单边换手率序列。
    """
    if isinstance(factor, pd.DataFrame):
        factor = factor.iloc[:, 0]

    def top_group_weights(cross_section: pd.Series) -> pd.Series:
        """为单日最高因子分组生成等权权重。

        输入为一个日期上所有股票的因子值，返回索引完全相同的权重
        Series。最高分位组等权，其余股票以及缺失样本权重为零。
        """
        values = cross_section.dropna()
        if len(values) < quantiles:
            return pd.Series(0.0, index=cross_section.index)
        groups = pd.qcut(values.rank(method="first"), quantiles, labels=False)
        selected = values[groups == groups.max()]
        weights = pd.Series(1.0 / len(selected), index=selected.index)
        return weights.reindex(cross_section.index).fillna(0.0)

    weights = factor.unstack().apply(top_group_weights, axis=1)
    turnover = weights.fillna(0.0).diff().abs().sum(axis=1) / 2.0
    turnover.name = "turnover"
    return turnover.dropna()


def calc_forward_returns(
    market_data: pd.DataFrame,
    period: int,
    price_col: str = "adj_close",
) -> pd.Series:
    """根据价格面板计算指定周期的未来持有收益。

    计算口径为在因子日后的第一个交易日价格买入，并在再向后 ``period``
    个交易日的价格卖出：``price[t+1+period] / price[t+1] - 1``。这种口径
    避免直接使用因子形成日收盘价成交。结果尾部因缺少未来价格会自然缺失。

    Parameters
    ----------
    market_data : pd.DataFrame
        索引为 ``(date, symbol)`` 的行情数据。
    period : int
        未来持有交易日数量，必须为正整数。
    price_col : str, default "adj_close"
        用于计算收益的价格列。

    Returns
    -------
    pd.Series
        ``(date, symbol)`` MultiIndex 收益 Series，名称为
        ``fwd_ret_<period>``。

    Raises
    ------
    ValueError
        ``period`` 不是正整数，或缺少指定价格列时抛出。
    """
    if period <= 0:
        raise ValueError("period 必须为正整数")
    if price_col not in market_data.columns:
        raise ValueError(f"market_data 缺少未来收益价格列: {price_col}")
    price = market_data[price_col].unstack()
    return (
        price.shift(-(1 + period)) / price.shift(-1) - 1
    ).stack().rename(f"fwd_ret_{period}")


def _timing_horizon_weights(
    horizon_weights: dict[int, float] | None = None,
) -> tuple[dict[int, float], float]:
    """校验并返回择时胜率周期权重。"""
    weights = horizon_weights or TIMING_WIN_HORIZON_WEIGHTS
    if not weights or any(horizon <= 0 for horizon in weights):
        raise ValueError("horizon_weights 必须包含正整数周期")
    total_weight = float(sum(weights.values()))
    if total_weight <= 0:
        raise ValueError("horizon_weights 权重和必须大于 0")
    return weights, total_weight


def calc_max_drawdown(returns: pd.Series) -> float:
    """根据周期收益序列计算复利净值的最大回撤。

    先删除缺失收益，再计算 ``cumprod(1 + returns)`` 得到财富曲线。回撤为
    当前财富相对历史峰值的跌幅，返回其中最小值，因此结果通常小于或等于
    零。空收益序列返回 0。

    Parameters
    ----------
    returns : pd.Series
        按时间排序的周期收益率。

    Returns
    -------
    float
        最大回撤，例如 ``-0.2`` 表示从峰值下跌 20%。
    """
    returns = returns.dropna()
    if returns.empty:
        return 0.0
    wealth = (1 + returns).cumprod()
    drawdown = (wealth - wealth.cummax()) / wealth.cummax()
    return float(drawdown.min())


def layered_backtest(
    factor: pd.DataFrame | pd.Series,
    returns: pd.DataFrame | pd.Series,
    n_groups: int = 5,
    annual_trading_days: int = 252,
    period: int = 1,
) -> LayeredResult:
    """按因子值进行截面分层，并汇总各层收益和风险指标。

    每个日期先对因子值排名，以 ``method="first"`` 打破相同因子值的并列，
    然后用 qcut 划分为 1 到 ``n_groups``。组号越大表示因子值越高，各组
    收益为组内股票的等权平均收益。函数进一步计算最高/最低因子组的累计
    收益、年化收益、年化夏普和最大回撤。

    Parameters
    ----------
    factor : pd.Series or pd.DataFrame
        ``(date, symbol)`` MultiIndex 因子值；DataFrame 只使用第一列。
    returns : pd.Series or pd.DataFrame
        与因子索引对齐的未来收益；DataFrame 只使用第一列。
    n_groups : int, default 5
        每个日期截面的分组数量。
    annual_trading_days : int, default 252
        基础数据每年的交易周期数量。
    period : int, default 1
        每条收益覆盖的周期数。年化频率为
        ``annual_trading_days / period``，收益本身不会被 period 除算。

    Returns
    -------
    LayeredResult
        包含逐期分组收益、累计收益及各类汇总指标。

    Raises
    ------
    ValueError
        ``period <= 0``，或输入未使用至少两层 MultiIndex 时抛出。

    Notes
    -----
    调用方应确保输入收益与实际调仓频率一致。若把每天都存在的重叠多日
    forward return 直接复利，会高估可交易组合的累计表现。
    """
    if period <= 0:
        raise ValueError("period 必须为正整数")
    if isinstance(factor, pd.DataFrame):
        factor = factor.iloc[:, 0]
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]

    combined = pd.DataFrame({"factor": factor, "returns": returns}).dropna()
    if not isinstance(combined.index, pd.MultiIndex) or combined.index.nlevels < 2:
        raise ValueError("factor 和 returns 必须使用 MultiIndex: (date, symbol)")

    records = []
    for current_date, cross_section in combined.groupby(level=0):
        cross_section = cross_section.droplevel(0)
        if len(cross_section) < n_groups:
            continue
        groups = pd.qcut(
            cross_section["factor"].rank(method="first"),
            n_groups,
            labels=False,
            duplicates="drop",
        ) + 1
        for group in range(1, n_groups + 1):
            group_returns = cross_section.loc[groups == group, "returns"]
            if not group_returns.empty:
                records.append((current_date, group, group_returns.mean()))

    if not records:
        return LayeredResult(
            group_returns=pd.DataFrame(),
            top_cumulative_returns=pd.Series(dtype=float),
            bottom_cumulative_returns=pd.Series(dtype=float),
            n_groups=n_groups,
        )

    group_returns = pd.DataFrame(
        records, columns=["date", "group", "returns"]
    ).pivot(index="date", columns="group", values="returns").sort_index()
    periods_per_year = annual_trading_days / period
    cumulative_returns = (1 + group_returns).cumprod() - 1
    total_returns = (1 + group_returns).prod(skipna=True)
    valid_periods = group_returns.notna().sum().clip(lower=1)
    annual_returns = total_returns.where(total_returns > 0) ** (
        periods_per_year / valid_periods
    ) - 1
    annual_returns.name = "annual_return"

    group_std = group_returns.std(ddof=1)
    sharpe_ratios = (
        group_returns.mean().divide(group_std.where(group_std > 0))
        * np.sqrt(periods_per_year)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sharpe_ratios.name = "sharpe_ratio"

    top_group = int(group_returns.columns.max())
    bottom_group = int(group_returns.columns.min())
    top_cumulative_returns = cumulative_returns[top_group].rename(
        "top_cumulative_returns"
    )
    bottom_cumulative_returns = cumulative_returns[bottom_group].rename(
        "bottom_cumulative_returns"
    )
    top_annual_return = float(annual_returns.loc[top_group])
    bottom_annual_return = float(annual_returns.loc[bottom_group])
    top_sharpe_ratio = float(sharpe_ratios.loc[top_group])
    bottom_sharpe_ratio = float(sharpe_ratios.loc[bottom_group])
    top_max_drawdown = calc_max_drawdown(group_returns[top_group])
    bottom_max_drawdown = calc_max_drawdown(group_returns[bottom_group])
    top_bottom_returns = (
        group_returns[top_group] - group_returns[bottom_group]
    ).dropna()
    win_rate = (
        float((top_bottom_returns > 0).mean())
        if not top_bottom_returns.empty
        else 0.0
    )

    return LayeredResult(
        group_returns=group_returns,
        top_cumulative_returns=top_cumulative_returns,
        bottom_cumulative_returns=bottom_cumulative_returns,
        top_annual_return=top_annual_return,
        bottom_annual_return=bottom_annual_return,
        top_sharpe_ratio=top_sharpe_ratio,
        bottom_sharpe_ratio=bottom_sharpe_ratio,
        top_max_drawdown=top_max_drawdown,
        bottom_max_drawdown=bottom_max_drawdown,
        win_rate=win_rate,
        n_groups=n_groups,
    )

def eval(
    factor_values: pd.DataFrame | pd.Series,
    market_data: pd.DataFrame,
    forward_period: int = 1,
    n_groups: int = 5,
    ic_method: str = "rank",
    price_col: str = "adj_close",
) -> FactorEvalResult:
    """汇总单个因子的 IC、换手率和分层回测指标。

    函数根据 ``market_data`` 计算指定周期的未来收益，并将因子值与收益
    对齐，汇总 IC、换手率、分层收益和风险指标。``IC_mean``、``IC_std``
    和 ``ICIR`` 使用完整 IC 序列的多 offset 汇总；组合日期采样确保多日
    未来收益不会因每日调仓而发生持有期重叠。DataFrame 因子输入应为
    ``date × symbol`` 宽表；Series 输入应使用 ``(date, symbol)``
    MultiIndex。

    Parameters
    ----------
    factor_values : pd.DataFrame or pd.Series
        因子计算结果。DataFrame 的索引为 date、列为 symbol；Series 的
        索引为 ``(date, symbol)``。
    market_data : pd.DataFrame
        ``(date, symbol)`` MultiIndex 行情数据，至少包含 ``price_col`` 列，
        默认是 ``adj_close``。
    forward_period : int, default 1
        未来收益持有周期和调仓间隔。函数每隔该数量的交易日选择一次
        因子截面，同时用于 IC offset 汇总和分层收益年化频率修正。
    n_groups : int, default 5
        每个日期截面的分组数量。
    ic_method : str, default "rank"
        ``"rank"`` 计算 Spearman RankIC；其他值计算 Pearson IC。
    max_lag : int, default 20
        IC 衰减曲线计算的最大滞后期。

    Returns
    -------
    FactorEvalResult
        完整评估结果，包含汇总表、完整 IC 序列、换手率和分层结果，
        可直接交给绘图模块使用。

    Raises
    ------
    ValueError
        ``forward_period`` 或 ``n_groups`` 不是正整数时抛出。
    """
    if forward_period <= 0:
        raise ValueError("forward_period 必须为正整数")
    if n_groups <= 0:
        raise ValueError("n_groups 必须为正整数")

    if isinstance(factor_values, pd.DataFrame):
        factor = factor_values.stack().rename("factor")
    else:
        factor = factor_values.rename("factor")

    forward_returns = calc_forward_returns(
        market_data,
        forward_period,
        price_col=price_col,
    )
    full_ic_series = calc_ic_series(factor, forward_returns, method=ic_method)
    offset_ic_stats = calc_offset_ic_stats(full_ic_series, period=forward_period)
    timing_score = forward_returns.ge(0).astype(float)
    full_timing_ic_series = calc_ic_series(
        factor,
        timing_score,
        method="rank",
    ).rename("TimingIC")
    timing_ic_stats = calc_offset_ic_stats(
        full_timing_ic_series,
        period=forward_period,
    )
    tail_group_returns = calc_tail_group_returns(factor, forward_returns)
    daily_metrics = pd.concat(
        [
            full_ic_series.rename("IC"),
            full_timing_ic_series.rename("TimingIC"),
            tail_group_returns,
        ],
        axis=1,
    ).reindex(columns=DAILY_METRIC_COLUMNS).sort_index()
    daily_metrics.index.name = "date"

    # sampled_dates 用于分层回测、尾部组合收益和换手率计算，确保每个调仓期的
    # 未来收益不重叠。
    trading_dates = pd.DatetimeIndex(
        market_data.index.get_level_values("date").unique()
    ).sort_values()
    sampled_dates = trading_dates[::forward_period]
    sampled_factor = factor[
        factor.index.get_level_values("date").isin(sampled_dates)
    ]
    sampled_forward_returns = forward_returns[
        forward_returns.index.get_level_values("date").isin(sampled_dates)
    ]
    sampled_ic_series = full_ic_series[full_ic_series.index.isin(sampled_dates)]
    turnover = calc_turnover(sampled_factor, quantiles=n_groups)
    layered = layered_backtest(
        sampled_factor,
        sampled_forward_returns,
        n_groups=n_groups,
        period=forward_period,
    )
    sampled_tail_group_returns = calc_tail_group_returns(
        sampled_factor,
        sampled_forward_returns,
        fractions=(0.05, 0.10, 0.15),
    )

    t_stat, p_value = calc_t_stat(full_ic_series, period=forward_period)

    def final_cumulative_return(returns: pd.Series) -> float:
        values = returns.dropna()
        return float(values.iloc[-1]) if not values.empty else 0.0

    def tail_cumulative_return(column: str) -> float:
        if column not in sampled_tail_group_returns:
            return 0.0
        returns = sampled_tail_group_returns[column].dropna()
        return float((1.0 + returns).prod() - 1.0) if not returns.empty else 0.0

    ic_mean = offset_ic_stats["IC_mean"]
    ic_half_life = calc_ic_half_life(full_ic_series, ic_mean)
    ic_positive_rate = calc_ic_positive_rate(full_ic_series)
    quintile_monotonicity = calc_quintile_monotonicity(
        layered.group_returns,
        ic_mean,
        n_groups=n_groups,
    )

    summary = pd.DataFrame([{
        "period": forward_period,
        "IC_mean": round(ic_mean, 4),
        "IC_std": round(offset_ic_stats["IC_std"], 4),
        "ICIR": round(offset_ic_stats["ICIR"], 4),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "timing_IC_mean": round(timing_ic_stats["IC_mean"], 4),
        "IC_half_life": round(ic_half_life, 4),
        "IC_positive_rate": round(ic_positive_rate, 4),
        "quintile_monotonicity": round(quintile_monotonicity, 4),
        "top_5%_cumulative_return": round(
            tail_cumulative_return("top_5%_return"),
            4,
        ),
        "bottom_5%_cumulative_return": round(
            tail_cumulative_return("bottom_5%_return"),
            4,
        ),
        "top_10%_cumulative_return": round(
            tail_cumulative_return("top_10%_return"),
            4,
        ),
        "bottom_10%_cumulative_return": round(
            tail_cumulative_return("bottom_10%_return"),
            4,
        ),
        "top_15%_cumulative_return": round(
            tail_cumulative_return("top_15%_return"),
            4,
        ),
        "bottom_15%_cumulative_return": round(
            tail_cumulative_return("bottom_15%_return"),
            4,
        ),
        "top_20%_cumulative_return": round(
            final_cumulative_return(layered.top_cumulative_returns),
            4,
        ),
        "bottom_20%_cumulative_return": round(
            final_cumulative_return(layered.bottom_cumulative_returns),
            4,
        ),
        "top_20%_annual_return": round(layered.top_annual_return, 4),
        "bottom_20%_annual_return": round(layered.bottom_annual_return, 4),
        "top_20%_sharpe_ratio": round(layered.top_sharpe_ratio, 4),
        "bottom_20%_sharpe_ratio": round(layered.bottom_sharpe_ratio, 4),
        "top_20%_max_drawdown": round(layered.top_max_drawdown, 4),
        "bottom_20%_max_drawdown": round(layered.bottom_max_drawdown, 4),
        "top_20%_bottom_20%_win_rate": round(layered.win_rate, 4),
    }])
    return FactorEvalResult(
        summary=summary.set_index("period"),
        full_ic_series=full_ic_series,
        daily_metrics=daily_metrics,
        sampled_ic_series=sampled_ic_series,
        turnover=turnover,
        layered=layered,
    )
