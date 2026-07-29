# MLFactors

MLFactors 是一个面向 A 股日频横截面研究的本地因子计算、评估和机器学习组合
项目。当前数据覆盖沪深300、中证500和中证1000三套历史股票池，主因子库包含
232 个价格、风险、财务和行业因子，并提供中证500上的 SVM、LightGBM、MLP
滚动组合因子流程。

项目有两条主要数据链路：

```text
单因子链路
cache/<index>_csv/*.csv
    -> DataLoader：字段规范化、历史成分股和停牌/ST过滤
    -> Factor.generate_signals()
    -> factors_eval.eval()
    -> outputs/.../<factor>/
    -> aggregate_factor_outputs.py
    -> factor_results_232/<index>_factor_wide/*.csv

机器学习链路
factor_results_232/zz500_factor_wide/*.csv
    + cache/zz500_csv/*.csv
    -> ml/convert_to_qlib.py
    -> ml/data/zz500/dataset.parquet
    -> SVM / LightGBM / MLP 按月滚动训练
    -> ml/<METHOD>/output/factors/*.csv
    -> ml/backtest.py
    -> ml/<METHOD>/output/backtest/*.csv
```

当前主要能力：

- 从按股票拆分的 A 股 CSV 构造 `(date, symbol)` 多级索引
  （MultiIndex）数据。
- 使用历史指数成分股，避免直接用当前成分股回测整个历史区间。
- 按日剔除停牌和 ST 股票，并归一部分历史证券代码。
- 通过注册表（registry）和 JSON 配置管理 232 个单因子。
- 计算 Rank IC、Timing IC、ICIR、换手率、分层收益、尾部组合收益、夏普比率、
  最大回撤和胜率等指标。
- 保存总时段和逐自然年的 CSV、图表及 Markdown 报告。
- 将单因子结果聚合为每只股票一张 232 因子宽表。
- 使用统一的数据、标签、滚动窗口和评估方式训练三类机器学习组合因子。

## 环境

优先使用项目根目录的虚拟环境：

```bash
./.venv/bin/python -c "import sys; print(sys.executable)"
```

当前主流程依赖：

- Python 3.10+
- pandas
- numpy
- scipy
- matplotlib
- tabulate，用于生成 Markdown 表格

可以用下面的命令检查核心依赖：

```bash
./.venv/bin/python -c "import pandas, numpy, scipy, matplotlib, tabulate"
```

项目直接从仓库根目录运行，不要求安装为 Python 包。

## A 股数据约定

### 三套指数缓存

当前本地数据目录如下：

| 数据目录 | 指数 | 指数代码 | 当前股票 CSV 数 | 数据起点 |
| --- | --- | --- | ---: | --- |
| `cache/hs300_csv/` | 沪深300 | `000300.SH` | 637 | 2016-01-29 |
| `cache/zz500_csv/` | 中证500 | `000905.SH` | 1653 | 2010-01-12 |
| `cache/zz1000_csv/` | 中证1000 | `000852.SH` | 2759 | 2015-06-30 |

股票 CSV 数是各指数历史股票池涉及过的证券总数，不是任一交易日的成分股数量。
当前股票行情缓存最晚到 2026-07-22。各目录还包含：

```text
<index>_csv/
├── 000001.SZ.csv
├── 600000.SH.csv
├── ...
└── constituents_daily.csv
```

`DataLoader` 读取 `constituents_daily.csv`，其列为：

| 字段 | 含义 |
| --- | --- |
| `date` | 交易日 |
| `tickers` | 当日成分股代码，以英文逗号分隔 |

加载器将逐日成分关系对齐到行情交易日，并只保留当日指数成员。
沪深300和中证1000缓存中可能仍保留旧的 `index_members_rebalance.csv`，
加载器只会跳过该文件，不再读取它作为成分股数据；中证500缓存中的旧文件已删除。

中证500的 `constituents_daily.csv` 覆盖 2010-01-12 至 2026-07-22，共4012个
交易日，每日严格500只。成员名单来自 `tmp/index_members_v2_202607291546.csv`
中的459个历史截面；截面之间的切换日期按真实生效日还原，而不是使用月末快照日：
2013年7月及以前的定期调整从1月、7月首个交易日起生效，2013年12月起从6月、
12月第二个星期五后的下一交易日起生效；退市、吸收合并、风险警示和证券代码
变更按相应事件的实际生效日切换。重建结果已逐一与459个原始截面核对一致。

### 股票 CSV

每个 `<股票代码>.csv` 对应一只 A 股证券。实际缓存包含约 124 列，主要分为：

| 类型 | 代表字段 | 用途 |
| --- | --- | --- |
| 标识与日期 | `ts_code`, `trade_date` | 股票代码和交易日 |
| 原始行情 | `open`, `high`, `low`, `close` | 未复权价格 |
| 复权行情 | `adj_open`, `adj_high`, `adj_low`, `adj_close`, `adj_factor` | 因子计算与回测 |
| 成交与流动性 | `volume`, `amount`, `turnover_rate`, `pct_chg` | 量价和流动性因子 |
| 市值与估值 | `market_cap`, `circ_mv`, `pe_ratio`, `pb_ratio`, `ps_ratio` | 市值、估值和风格因子 |
| 股票属性 | `stock_name`, `board`, `sector`, `industry_1`, `industry_2` | 板块和行业截面 |
| 指数行情 | `hs300_close`, `zz500_close`, `zz1000_close` | 回测基准 |
| 财报时点 | `report_date`, `publish_date`, `ann_date_*` | 财报期和可用时间 |
| 财务数据 | `total_assets`, `revenue`, `net_income`, `eps` 等 | 财务类因子 |

加载时会执行以下规范化：

- `trade_date` 统一改名为内部字段 `date`。
- 返回数据使用 `(date, symbol)` MultiIndex，并按日期和股票排序。
- `gross_margin`、`operating_cash_flow`、`sector` 等统一字段会从实际源字段映射。
- `gross_profit`、`vwap`、`non_current_debt` 等字段可由源字段派生。
- 历史代码通过 `stock_basic_symbol` 和内置别名映射归并到现行代码，避免同一证券
  被拆成两个 symbol。
- 只读取因子配置中声明的字段，并自动补充回测价格 `adj_close`；财务类因子还会
  补充 `publish_date`。

`dataloader.py` 当前包含四条显式的历史代码到现行代码映射：

| 历史代码 | 现行代码 |
| --- | --- |
| `000022.SZ` | `001872.SZ` |
| `000043.SZ` | `001914.SZ` |
| `300114.SZ` | `302132.SZ` |
| `601313.SH` | `601360.SH` |

加载股票文件、指数成分股和停牌/ST状态时都会应用同一映射。除此之外，加载器
还会读取每只股票 CSV 的 `stock_basic_symbol`，自动推导文件代码到现行代码的
映射；上述四条内置记录用于覆盖仅有现行代码文件、但历史成分股记录仍使用旧
代码等无法完全依赖 CSV 自动识别的情况。归一后，同一证券的新旧代码数据会按
`(date, symbol)` 去重，优先保留文件名已经是现行代码的数据。

财务数据的报告期和发布时间已经随行情日对齐在股票 CSV 中。主流程直接使用
缓存中的时点数据，不会在运行时重新推导财报发布时间。生成或更新缓存时必须
继续保证 `publish_date` 不晚于对应信号日，避免前视偏差（look-ahead bias）。

### 历史成分股、状态过滤和基准

传入 `constituents_path` 与 `constituent_index` 后，加载器将成分关系对齐到行情
交易日并前向填充，再只保留当日指数成员。两个参数必须同时提供或同时不提供。

当前三套指数的参数为：

| 指数 | `data_dir` | `constituent_index` | 推荐起点 |
| --- | --- | --- | --- |
| 沪深300 | `cache/hs300_csv` | `000300.SH` | `2016-01-29` |
| 中证500 | `cache/zz500_csv` | `000905.SH` | `2010-01-12` |
| 中证1000 | `cache/zz1000_csv` | `000852.SH` | `2015-06-30` |

停牌和 ST 过滤默认使用：

```text
cache/tushare_security_status_missing_tables_20100104_20260623_20260706.csv
```

其中逐日停牌记录与 ST 起止区间会转换成布尔状态表，命中的
`(date, symbol)` 行会被删除。设置 `security_status_path=None` 可关闭过滤。

基准列会规范成 `HS300`、`ZZ500`、`ZZ1000`，用于评估图中的基准累计收益。
代码仍保留 `SPY_close` 和 `QQQ_close` 兼容别名，但当前 A 股缓存和主流程使用
三大指数基准，不再以 SPY、QQQ 为研究对象。

## 因子库与配置

### 232 因子

主配置是 `config/factor_configs_232.json`，每个条目包含：

```json
{
  "BETA_10d": {
    "module": "factors.price.alpha158",
    "relay_class": "Alpha158PriceFactor",
    "params": {
      "operation": "BETA",
      "window": 10
    },
    "columns": [
      "adj_close"
    ]
  }
}
```

字段含义：

- `module`：运行前需要动态导入的 Python 模块。
- `relay_class`：注册表中的类名。当前注册表以类名为键，不以因子的
  `name` 属性为键。
- `params`：构造因子实例时传入的参数。
- `columns`：从股票缓存中读取的最小字段集合。

232 个因子的目录分类为：

| 类别 | 数量 | 说明 |
| --- | ---: | --- |
| `price` | 154 | Alpha158 量价因子及反转、成交、量价相关等扩展因子 |
| `risk` | 30 | Alpha158 风险因子、Beta、半 Beta、波动率和规模等 |
| `fundamental` | 45 | 资产负债、盈利、成长、现金流和估值类财务因子 |
| `sector` | 3 | 行业动量、风格聚类和行业财务稳定性因子 |

其中 142 个价格 Alpha158 因子与 16 个风险 Alpha158 因子合计 158 个，另外
74 个为扩展因子。`config/factor_configs_69.json` 是当前 232 因子名称集合的
69 因子子集，部分参数与 232 主配置独立设置。

### 财务类名单

`config/financial_factors_232.json` 单独记录 46 个需要输出财报年龄的因子：

- 45 个 `factors.fundamental.*` 因子。
- `industry_current_asset_growth_stability_24m` 虽位于 `sector` 目录，但直接
  使用 `current_assets`，因此同样按财务类处理。

仅使用行情、成交、市值、行业标签或每日估值字段的因子不在该名单中。

### 因子实现

所有因子继承 `factors.base.BaseFactor`，并通过
`factors.registry.register_factor` 注册。`generate_signals()` 接收
`(date, symbol)` MultiIndex 数据，返回“日期 × 股票”的因子宽表：

```text
index   = DatetimeIndex，名称为 date
columns = 股票代码，名称为 symbol
values  = 当日因子值
```

因子实现按 `factors/price`、`factors/risk`、`factors/fundamental`、
`factors/sector` 分类。`factors/router.py` 可把已经合并进输入数据的某一列直接
路由为信号矩阵。

## 单因子计算与回测

### 命令行运行

当前 `run.py` 命令行入口固定使用中证500：

- 数据目录：`cache/zz500_csv`
- 成分指数：`000905.SH`
- 开始日期：`2010-01-01`
- 输出目录：`outputs/zz500/review/<因子名>`
- 默认 5 日前向收益、5 组分层和 Rank IC

当前工作区不存在默认参数指向的 `config/factor_configs.json`，因此运行时应
显式指定配置文件。推荐工作目录是项目根目录：

```bash
cd /data_all/dyj/MLFactors
```

运行一个因子并同时保存因子值：

```bash
/data_all/dyj/miniconda3/envs/fshc/bin/python run.py \
  --factor-config config/factor_configs_232.json \
  --factor BETA_10d \
  --save-factor
```

不传 `--factor` 会顺序运行配置中的全部因子；不传 `--save-factor` 仍会回测并
生成报告，但不会创建按股票拆分的 `factors/*.csv`。

### 在代码中运行三大指数

命令行入口尚未提供指数选择参数。沪深300和中证1000需要直接构造 `Runner`，
中证500也可以使用同样方式自定义区间和输出目录：

```python
import importlib
import json
from pathlib import Path

from run import Runner


factor_name = "BETA_10d"
factor_configs = json.loads(
    Path("config/factor_configs_232.json").read_text(encoding="utf-8")
)
financial_names = set(
    json.loads(
        Path("config/financial_factors_232.json").read_text(encoding="utf-8")
    )["factor_names"]
)
config = factor_configs[factor_name]
importlib.import_module(config["module"])

data_dir = Path("cache/hs300_csv")
runner = Runner(
    factor_name=factor_name,
    relay_class=config["relay_class"],
    factor_params=config.get("params", {}),
    start="2016-01-29",
    data_columns=config["columns"],
    data_dir=data_dir,
    constituents_path=data_dir / "constituents_daily.csv",
    constituents="000300.SH",
    output_dir=Path("outputs/hs300") / factor_name,
    is_financial_factor=factor_name in financial_names,
)
runner.run(save_factor=True)
```

将 `data_dir`、`constituents` 和 `start` 替换为前述参数即可运行另外两个指数。

## 评估口径

### 前向收益和可交易时间

默认评估周期为 5 个交易日，使用复权收盘价：

```text
forward_return[t] = adj_close[t + 1 + period] / adj_close[t + 1] - 1
```

因子在 `t` 日形成，下一真实交易日 `t+1` 进入，持有 `period` 个交易日。
因此 5 日收益是 `adj_close[t+6] / adj_close[t+1] - 1`。

### 指标

总时段与逐年回测均重新计算以下指标：

- Rank IC 的均值、标准差、ICIR、t 统计量、p 值、正率和 IC 半衰期。
- Timing IC 均值。
- 5 组收益及五分组单调性。
- 因子最高 20% 组合的单边换手率。
- 顶部和底部 5%、10%、15%、20%、25%、30% 组合的累计收益及最大回撤。
- 顶部和底部 20% 组合的夏普比率，以及顶部相对底部胜率。
- 日频 IC、Timing IC 和上述顶部、底部比例组合的未来收益。

多日收益的分层回测、尾部组合和换手率按每 `period` 个交易日采样，避免直接把
重叠的多日收益逐日连续复利。IC 统计会按不同交易日起始偏移分别采样后汇总，
降低单一起始日造成的偏差。

`IC_half_life` 使用中心化后的 5 日滚动 IC 均值，寻找未来首次连续两日低于
当前有效性一半的位置，并按交易日求平均。负向有效因子会先按 IC 均值翻转方向。

`quintile_monotonicity` 位于 `[-1, 1]`：

- `+1` 表示五组收益完全沿 IC 均值方向单调。
- `0` 表示没有稳定单调关系。
- 负值表示分层收益方向与 IC 均值相反。

年度报告会把行情和信号切在自然年内部重新计算，因此年末收益不会借用下一年
价格。单因子评估图当前是 2×2 四面板：IC 时序、分层累计收益、换手率和汇总表。

## 单因子输出与聚合

### Runner 输出

`runner.run(save_factor=True)` 的标准输出结构为：

```text
<output_dir>/
├── report.md
├── factor_summary.csv
├── daily_factor_metrics.csv
├── <factor>_5d.png
├── yearly_summary.csv
├── yearly_factor_summary_trends.png
├── yearly/
│   └── <year>/
│       ├── factor_summary.csv
│       ├── <factor>_5d.png
│       └── report.md
└── factors/
    ├── 000001.SZ.csv
    ├── 600000.SH.csv
    └── ...
```

普通因子的每只股票 CSV 为：

```text
signal_date,available_date,<因子名>
```

财务类因子为：

```text
signal_date,available_date,<因子名>,days_since_latest_report_publish_date
```

- `signal_date` 是因子形成日，即原有内部字段 `date`。
- `available_date` 是加载交易日历中 `signal_date` 的下一真实交易日。
- `days_since_latest_report_publish_date` 是信号日距当时最近财报
  `publish_date` 的自然日天数。
- 首次有效财报发布之前，财报年龄为空。
- 数据集最后一个交易日若没有可验证的下一交易日，该行不会写入单因子 CSV。

当前本地 `outputs/zz500/<因子名>` 下已有 232 个历史批处理结果目录；它们与
现有 `run.py` 命令行使用的 `outputs/zz500/review/<因子名>` 不是同一路径。
历史目录中的 `.complete.json`、`market_state_summary.csv` 等文件来自此前的
批处理过程，不是当前 `Runner.run()` 的标准输出。

### 聚合为股票宽表

`aggregate_factor_outputs.py` 将多个单因子目录聚合成“每只股票一张宽表”。
读取时兼容旧的 `date,<factor>` 和新的三列、四列单因子格式；聚合结果只保留
`date` 与因子值，不把 `available_date` 和财报年龄重复写入 232 因子宽表。

聚合当前 `run.py` 生成的中证500结果：

```bash
/data_all/dyj/miniconda3/envs/fshc/bin/python aggregate_factor_outputs.py \
  --outputs-dir outputs/zz500/review \
  --output-dir factor_results_232/zz500_factor_wide
```

聚合现有历史批处理目录时，将 `--outputs-dir` 改为 `outputs/zz500`。

当前本地 `factor_results_232/zz500_factor_wide` 有 1651 只股票，每个 CSV
首列是 `date`，随后是严格相同顺序的 232 个因子列。再次聚合时，同名因子列会
被新结果替换，其他已有因子列会保留。

## 机器学习组合因子

`ml/` 当前实现中证500上的 232 因子滚动机器学习组合。详细的模型专用说明见
`ml/README.md`，以下说明它与主项目的衔接方式。

### 数据转换

`ml/convert_to_qlib.py` 的默认输入为：

```text
factor_results_232/zz500_factor_wide/*.csv
cache/zz500_csv/*.csv
```

转换器会：

1. 要求所有因子宽表表头一致、首列为 `date`，且恰好有 232 个因子。
2. 按股票合并 `adj_close` 和 `zz500_close`。
3. 计算个股未来 5 日收益和相对中证500的未来 5 日超额收益。
4. 保留原始因子缺失值，不在 Parquet 中提前填充。
5. 按全市场因子非空覆盖率寻找首个可训练日期。
6. 生成 Qlib 分组列：`feature`、`label` 和 `meta`。
7. 原子替换 `dataset.parquet`，并写入 `manifest.json`。

输出为：

```text
ml/data/zz500/
├── dataset.parquet
└── manifest.json
```

当前本地 manifest 记录：

| 项目 | 当前值 |
| --- | ---: |
| 股票文件数 | 1651 |
| 样本行数 | 1,948,551 |
| 因子数 | 232 |
| 日期范围 | 2010-01-12 至 2026-07-22 |
| 覆盖率阈值 | 30% |
| Qlib 版本 | 0.9.7 |

标签公式为：

```text
fwd_return_5d = adj_close[t+6] / adj_close[t+1] - 1
fwd_excess_return_5d = fwd_return_5d - zz500_forward_return_5d
```

转换命令：

```bash
/data_all/dyj/miniconda3/envs/fshc/bin/python ml/convert_to_qlib.py
```

可通过 `--source-dir`、`--cache-dir`、`--output` 和
`--coverage-threshold` 覆盖默认路径和阈值。

### 统一预处理与标签

三种模型共享 `ml/common.py`：

- 每个交易日对每个因子做 5 倍 MAD 去极值。
- 每个交易日对每个因子做横截面 Z-score 标准化。
- `NaN` 和无穷值只在模型运行内存中填 0，原始宽表和 Parquet 不改写。
- 按每日未来 5 日超额收益排序，前 30% 标记为 `+1`，后 30% 标记为 `-1`，
  中间 40% 不进入分类损失。

这里的“前 30%”指未来超额收益最高的一组，“后 30%”指最低的一组。

### 滚动训练

对每个预测月 `M`：

```text
扩展训练集：首个合格日期 -> M-13 月月末
固定测试集：M-12 月月初 -> M-1 月月末，共12个月
预测集：    M 月月初 -> M 月月末
```

训练集和测试集各清除最后 `horizon + 1 = 6` 个信号日，保证标签价格不越过
当前数据段边界。每个滚动窗口都重新初始化模型并训练 10 轮，不继承上个月参数。

每轮记录训练集和测试集的：

- 分类损失（loss）。
- IC 和 ICIR。
- Top 20% 阶段总收益。
- Top 20% 夏普比率。

Top 20% 阶段收益将每日最高 20% 股票等权组成 5 日组合，按 5 种交易日起始
偏移构造不重叠复利路径，再平均五条路径的期末总收益。每月选择测试集
Top 20% 阶段收益最高的一轮；若收益相同，则选择测试损失更低的一轮，用该模型
预测月份 `M`。

### 三种模型

| 方法 | 实现 | 损失和主要设置 | 模型文件 |
| --- | --- | --- | --- |
| SVM | `SGDClassifier` 线性软间隔 SVM | Hinge loss，L2，逐轮 `partial_fit` | `latest.joblib` |
| LightGBM | GBDT 二分类 | Binary log loss，15 leaves，最大深度5，32线程 | `latest.txt` |
| MLP | 232→512→256→128→64→1 | LayerNorm、GELU、Dropout、AdamW、标签平滑 | `latest.pt` |

MLP 使用混合精度和余弦学习率，代码要求 CUDA；GPU 不可用时会直接报错，不会
回退到 CPU。SVM 和 LightGBM 使用 CPU。

训练命令：

```bash
/data_all/dyj/miniconda3/envs/fshc/bin/python ml/SVM/train.py
/data_all/dyj/miniconda3/envs/fshc/bin/python ml/LightGBM/train.py
/data_all/dyj/miniconda3/envs/fshc/bin/python ml/MLP/train.py --gpu-id 0
```

各模型参数位于对应的 `config.yaml`。当前三个配置均从 2020-01 开始输出，
并把数据集最后信号日 2026-07-22 的 `available_date` 显式设置为
2026-07-23。数据集更新后必须同步检查 `last_available_date`。

每次重新开始训练时会清理该方法旧的模型、训练日志和
`output/factors`，因此需要保留历史产物时应先复制到其他目录。

### 模型输出

三种方法使用相同目录结构：

```text
ml/<METHOD>/
├── config.yaml
├── train.py
├── model/
│   └── latest.{joblib|txt|pt}
├── log/
│   └── epochs.log
└── output/
    ├── factors/
    │   ├── 000001.SZ.csv
    │   └── ...
    └── backtest/
        ├── factor_summary.csv
        ├── yearly_summary.csv
        └── daily_factor_metrics.csv
```

每只股票的模型因子 CSV 为：

```text
signal_date,available_date,<METHOD>
```

当前 SVM、LightGBM、MLP 各有 1028 只股票的模型因子文件，信号日期覆盖
2020-01-02 至 2026-07-22。不同股票只包含其进入有效中证500股票池期间的行。

### 机器学习因子回测

`ml/backtest.py` 默认读取 `ml/<METHOD>/output/factors`，重新加载中证500历史
成分股、复权行情以及停牌/ST状态。总时段默认从 2020-01-01 到因子最后日期，
并对每个自然年独立切片重新计算，不借用下一年价格。

除单因子通用指标外，机器学习回测增加两个双向 Top 20% 排名效率：

- `top_20%_selection_rank_efficiency`：因子 Top 20% 股票在真实收益降序排名中
  的效率。
- `top_20%_return_capture_rank_efficiency`：真实收益 Top 20% 股票在因子降序
  排名中的捕获效率。

当每日有效股票数为 `n`、选择数为 `k = ceil(20% × n)` 时，理想排名和为
`k(k+1)/2`。效率等于理想排名和除以实际名次和，范围为 `(0, 1]`，完全匹配
为 1，随机排序的理论基准约为 0.2。

回测命令：

```bash
/data_all/dyj/miniconda3/envs/fshc/bin/python ml/backtest.py SVM
/data_all/dyj/miniconda3/envs/fshc/bin/python ml/backtest.py LightGBM
/data_all/dyj/miniconda3/envs/fshc/bin/python ml/backtest.py MLP
```

新增方法只要输出目录满足 `ml/<METHOD>/output/factors`，且因子值列名为
`<METHOD>`，即可使用同一回测入口。也可用 `--factor-column`、
`--factor-dir`、`--output-dir`、`--start`、`--end` 显式覆盖。

## 自定义因子

新因子需要继承 `BaseFactor`、定义 `name` 和 `description`，并注册类：

```python
import pandas as pd

from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class Momentum20(BaseFactor):
    name = "momentum_20d"
    description = "20日复权价格动量"

    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        close = data["adj_close"].unstack("symbol").astype(float)
        result = close.pct_change(20, fill_method=None)
        result.index.name = "date"
        result.columns.name = "symbol"
        return result
```

将文件保存为 `factors/price/momentum_20d.py`，然后在配置中增加：

```json
{
  "momentum_20d": {
    "module": "factors.price.momentum_20d",
    "relay_class": "Momentum20",
    "params": {},
    "columns": [
      "adj_close"
    ]
  }
}
```

注册表使用类名：

```python
import factors.price.momentum_20d

from factors.registry import FactorRegistry


factor_class = FactorRegistry.get("Momentum20")
print(FactorRegistry.list())
```

若新因子直接依赖财报字段并需要输出财报年龄，还应把配置中的因子名称加入
`config/financial_factors_232.json`。

## 项目结构

```text
MLFactors/
├── cache/
│   ├── hs300_csv/                    # 沪深300历史股票与成分关系
│   ├── zz500_csv/                    # 中证500历史股票与成分关系
│   ├── zz1000_csv/                   # 中证1000历史股票与成分关系
│   └── tushare_security_status*.csv  # 停牌、ST和上市状态记录
├── config/
│   ├── factor_configs_232.json       # 232因子主配置
│   ├── factor_configs_69.json        # 69因子子集配置
│   └── financial_factors_232.json    # 46个财务类因子名单
├── factors/
│   ├── base.py
│   ├── registry.py
│   ├── router.py
│   ├── price/
│   ├── risk/
│   ├── fundamental/
│   └── sector/
├── factor_results_232/
│   └── zz500_factor_wide/            # 每只股票一张232因子宽表
├── ml/
│   ├── data/zz500/                   # Qlib Parquet和manifest
│   ├── SVM/
│   ├── LightGBM/
│   ├── MLP/
│   ├── common.py
│   ├── convert_to_qlib.py
│   └── backtest.py
├── outputs/                          # 本地单因子与回测产物
├── scheme/
│   └── MaxICWeight.py                # 最大IC权重研究代码
├── aggregate_factor_outputs.py       # 单因子转股票宽表
├── dataloader.py                     # 行情、成分股、状态与基准加载
├── factors_eval.py                   # 因子评估与分层回测
├── plot.py                           # 评估图和年度趋势图
└── run.py                            # 通用Runner与中证500命令行入口
```

`meta_data/` 中仍保留美股字段、S&P 500 和 Russell 1000 的旧研究资料，但当前
A 股主流程不读取这些文件。根目录的
`zz500_lightgbm_prediction_4class_signals.csv` 也没有被当前源码引用，应视为
独立实验产物，而不是 `ml/LightGBM` 三列组合因子流程的输入或输出。

`scheme/MaxICWeight.py` 当前仍导入已不存在的 `scheme/neutralization.py`，
相关运行配置和入口也不在当前工作区，因此该目录属于未完成的研究代码，不能
作为现有可运行主流程。

## 注意事项

- `run.py` 当前只把中证500参数暴露在命令行中；三套缓存存在不等于三大指数均有
  完整命令行批处理入口。
- 中证500逐日成分文件最晚到 2026-07-22；其最近一次定期调整已从
  2026-06-15起生效。研究更晚区间前仍需同步更新成分名单和行情交易日。
- 默认证券状态文件的观测截止日为 2026-06-23。状态表缺失的后续日期会按未停牌、
  非 ST 处理，最新区间回测前应更新状态数据。
- 不传历史成分股文件时会使用加载到的全部股票，可能产生严重的幸存者偏差
  （survivorship bias）。
- `symbols` 只加载少量股票时，全市场交易日历可能退化为这些股票实际有行情的
  日期；生产输出建议使用完整指数股票池。
- 因子值越高对应更高分组，但不表示所有因子都应做多最高组，方向应结合 IC 和
  因子定义判断。
- 当前回测不包含交易成本、滑点、涨跌停成交约束、容量限制和真实撮合。
- `cache/`、`outputs/`、Parquet 和多数大型结果目录属于本地产物，不应假定它们
  会随 Git 仓库分发。
- 机器学习训练会覆盖同方法的旧模型、日志和因子输出；重新训练前应确认是否需要
  归档。

## License

MIT
