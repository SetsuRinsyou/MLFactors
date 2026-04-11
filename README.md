# MLFactors — 机器学习因子挖掘框架

通用 ML 因子挖掘框架，支持多数据源加载、插件式因子定义、全流程评估（IC/ICIR/分层回测）、通过 qlib 接入组合回测。

## 环境依赖

```bash
# 推荐使用 uv 创建虚拟环境
uv venv && source .venv/bin/activate

# 核心依赖
uv pip install pandas numpy scipy scikit-learn matplotlib pyyaml loguru

# AKShare 数据源（可选）
uv pip install akshare

# 树模型 LightGBM / XGBoost（可选）
uv pip install lightgbm xgboost

# PyTorch 神经网络（可选）
uv pip install torch

# qlib 组合回测（可选）
uv pip install pyqlib
```

所有代码平铺在项目根目录，直接 `cd` 到项目根目录后运行即可，**无需安装为 Python 包**。

## 项目结构

```
MLFactors/
├── config.py                   # YAML 配置加载（load_config / get_config）
├── config/
│   └── default.yaml            # 默认配置：数据路径、评估参数等
├── data/
│   ├── schema.py               # 统一列名枚举（Col / FundamentalCol）
│   ├── base.py                 # DataLoader 抽象基类
│   ├── local_loader.py         # 本地文件加载（CSV / Parquet / SQLite）
│   └── akshare_loader.py       # AKShare 数据源（含本地缓存）
├── factors/
│   ├── base.py                 # BaseFactor 抽象基类
│   ├── registry.py             # FactorRegistry + @register_factor 装饰器
│   └── library/                # 内置因子
│       ├── momentum.py         # momentum_5 / momentum_10 / momentum_20
│       └── volatility.py       # volatility_5 / volatility_20 / highlow_spread_20
├── models/
│   ├── base.py                 # BaseModel 抽象基类（含 TimeSeriesSplit 交叉验证）
│   ├── tree.py                 # TreeModel — LightGBM / XGBoost
│   ├── linear.py               # LinearModel — Ridge / Lasso
│   └── nn.py                   # NNModel — PyTorch MLP（含 early stopping）
├── evaluation/
│   ├── ic.py                   # calc_ic / calc_ic_series / calc_icir / calc_ic_decay / calc_turnover / calc_t_stat
│   ├── layered.py              # layered_backtest — 分 N 组回测 + 多空对冲
│   ├── plot.py                 # IC 序列图 / 分布图 / 分层净值曲线 / IC 衰减图
│   └── report.py               # FactorReport — 汇总全部指标
├── backtest/
│   └── qlib_adapter.py         # QlibAdapter — qlib TopkDropout 策略回测
├── pipeline/
│   └── runner.py               # FactorPipeline — 端到端流水线
└── tests/
    ├── test_data_loader.py
    ├── test_factors.py
    └── test_evaluation.py
```

## 快速上手

所有示例均在项目根目录下运行（保证 `import data`, `import factors` 等可以找到模块）。

### 1. 使用本地数据 + 内置因子评估

```python
from data import LocalLoader
from pipeline.runner import FactorPipeline

loader = LocalLoader(market_path="path/to/market.csv")

pipeline = FactorPipeline()
pipeline.set_data_loader(loader)
pipeline.add_factors(["momentum_5", "momentum_20", "volatility_20"])

reports = pipeline.run()           # 返回 dict[factor_name, FactorReport]
reports["momentum_5"].summary()    # 各周期 IC/ICIR/分层指标汇总表
```

### 2. 自定义因子（Plugin 式）

```python
from data.schema import Col
from factors import BaseFactor, register_factor

@register_factor
class MyAlphaFactor(BaseFactor):
    name = "my_alpha"
    description = "量价背离因子"
    category = "custom"

    def compute(self, market_data, fundamental_data=None):
        close = market_data[Col.CLOSE].unstack(Col.SYMBOL)
        volume = market_data[Col.VOLUME].unstack(Col.SYMBOL)
        ret = close.pct_change(5)
        vol_chg = volume.pct_change(5)
        factor = ret.rolling(20).corr(vol_chg)
        return factor.stack().rename(self.name)

# 注册后即可在 pipeline 中使用
pipeline.add_factors(["my_alpha"])
```

### 3. ML 模型模式

```python
from data import LocalLoader
from models.tree import TreeModel
from pipeline.runner import FactorPipeline

pipeline = FactorPipeline()
pipeline.set_data_loader(LocalLoader(market_path="path/to/market.csv"))
pipeline.add_factors(["momentum_5", "momentum_10", "momentum_20",
                       "volatility_5", "volatility_20", "highlow_spread_20"])
pipeline.set_model(TreeModel(engine="lgbm"))

result = pipeline.run()
# result["model_report"]       — 模型预测值的 FactorReport
# result["cv_result"]          — TimeSeriesSplit 交叉验证 IC
# result["feature_importance"] — 特征重要性 pd.Series
```

### 4. 查看所有内置因子

```python
from factors.registry import FactorRegistry

print(FactorRegistry.list())
# ['highlow_spread_20', 'momentum_10', 'momentum_20', 'momentum_5',
#  'volatility_20', 'volatility_5']

for d in FactorRegistry.list_detail():
    print(d["name"], d["category"], d["description"])
```

### 5. 单独使用评估模块

```python
from evaluation.ic import calc_ic_series, calc_icir, calc_forward_returns
from evaluation.layered import layered_backtest
from evaluation.plot import plot_factor_report

fwd = calc_forward_returns(market_data, periods=[1, 5, 20])
ic_s = calc_ic_series(factor_values, fwd[5])
print("ICIR:", calc_icir(ic_s))

result = layered_backtest(factor_values, fwd[5], n_groups=5)
print("多空年化:", result.long_short_annual)

plot_factor_report(ic_s, result, factor_name="momentum_5")
```

## 数据格式

所有数据加载器输出 `MultiIndex(date, symbol)` 格式的 DataFrame，列名使用 `data.schema.Col` 枚举。

**行情数据必填列：**

| `Col` 属性 | 列名 | 说明 |
|---|---|---|
| `Col.DATE` | `date` | 日期 |
| `Col.SYMBOL` | `symbol` | 股票代码 |
| `Col.OPEN` | `open` | 开盘价 |
| `Col.HIGH` | `high` | 最高价 |
| `Col.LOW` | `low` | 最低价 |
| `Col.CLOSE` | `close` | 收盘价 |
| `Col.VOLUME` | `volume` | 成交量 |

**列名不匹配时通过 `column_mapping` 映射：**

```python
loader = LocalLoader(
    market_path="data.csv",
    column_mapping={"Date": "date", "Code": "symbol", "Close": "close"},
)
```

**自定义数据源：** 继承 `DataLoader`，实现 `load_market_data()` 方法即可。

## 因子开发规范

| 要素 | 说明 |
|---|---|
| 继承 | `BaseFactor` |
| 装饰器 | `@register_factor` |
| `name` | 唯一字符串，不可重复 |
| `compute()` 输入 | `market_data: pd.DataFrame`（MultiIndex） |
| `compute()` 输出 | `pd.Series`，索引为 `MultiIndex(date, symbol)` |

## 评估指标说明

| 指标 | 函数 | 说明 |
|---|---|---|
| IC | `calc_ic` | 单期截面 Spearman / Pearson 相关系数 |
| IC 序列 | `calc_ic_series` | 逐期 IC 时间序列 |
| ICIR | `calc_icir` | IC 均值 / IC 标准差 |
| IC 衰减 | `calc_ic_decay` | 不同滞后期的均值 IC |
| t 统计量 | `calc_t_stat` | IC 序列对零的单样本 t 检验 |
| 换手率 | `calc_turnover` | 相邻期因子排名变化比例 |
| 分层回测 | `layered_backtest` | 分 N 组年化收益 / 夏普 / 多空对冲 |

## 配置

修改 `config/default.yaml` 或传入自定义配置文件：

```python
from config import load_config
cfg = load_config("my_config.yaml")   # 会与默认配置深度合并
```

可配置项：数据缓存路径、额外因子搜索目录、前向收益周期、IC 计算方法、分组数、回测日期范围等。

## 运行测试

```bash
# 安装测试依赖
uv pip install pytest pytest-cov

# 运行全部测试
pytest tests/ -v

# 带覆盖率
pytest tests/ -v --cov=. --cov-report=term-missing
```

## License

MIT
