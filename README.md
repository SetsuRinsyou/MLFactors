# MLFactors

MLFactors 是中证500日频横截面因子计算与单因子评估框架。当前正式因子集合为：

- 199个基础因子；
- 2,388个单基础因子数学变换；
- 3,360个多基础因子组合；
- 合计5,947个因子，每个因子均保存原始值（Raw factor value）和中性化值
  （Neutralized factor value）。

本文只说明当前中证500正式单因子流水线。`industry/`、`ml/`、`scheme/` 等其他
研究模块有各自的入口和说明，不属于本流水线。

## 一、正式全流程

正式全量入口只有 `run.py --pipeline`：

```text
cache/zz500_csv
  → 计算199个基础因子值，不生成报告
  → 在 .runtime/<run-id>/base_matrices 建立项目内临时矩阵
  → 计算5,748个派生因子值，不生成报告
  → 立即删除基础因子临时矩阵
  → 回填时间窗口相关因子的已有空值
  → 从最终落盘因子值统一生成5,947份 factor_report.md
  → 清理本次 .runtime/<run-id>
```

因此，正式流程不会在缺失值回填前生成报告，也不需要先执行宽表聚合脚本。
基础矩阵只存在于 `/data_all/duyj/MLFactors/.runtime`，不会在 `/tmp` 建立；正常
结束和可被Python捕获的异常都会进入清理逻辑。强制断电或 `kill -9` 无法执行清理，
此时可在确认没有运行中的流水线后手动删除遗留的 `.runtime/factor_pipeline_*`。

当前配置识别出168个含时间窗口的基础因子。连同直接使用这些基础因子的派生
因子，共5,446个因子进入回填阶段。其余因子不执行回填。

## 二、运行环境与输入文件

推荐工作目录：

```bash
cd /data_all/duyj/MLFactors
```

当前正式命令使用以下Python解释器：

```text
/data_all/duyj/miniconda3/envs/tmp/bin/python
```

核心Python依赖包括 `numpy`、`pandas`、`scipy`、`statsmodels` 和 `tqdm`。
可先检查环境：

```bash
/data_all/duyj/miniconda3/envs/tmp/bin/python -c \
'import numpy, pandas, scipy, statsmodels, tqdm; print("环境检查通过")'
```

正式全流程需要以下输入：

| 路径 | 用途 |
|---|---|
| `cache/zz500_csv/*.csv` | 个股行情、财务、市值、行业和中证500基准数据 |
| `cache/zz500_csv/constituents_daily.csv` | 每日中证500历史成分股 |
| `cache/tushare_security_status_missing_tables_20100104_20260623_20260706.csv` | ST和停牌过滤 |
| `zz500_lightgbm_prediction_4class_signals.csv` | 报告中的四类市场状态 |
| `config/factor_configs_5947.json` | 5,947个因子的完整计算定义 |
| `config/factor_categories_5947.json` | 5,947个因子的固定类别 |
| `config/financial_factors_in_199.json` | 21个基础财务因子名单及财报日期字段口径 |

`--start` 只控制起始日。正式CLI没有单独的 `--end` 参数，结束日取输入缓存中
可用的最后交易日；未来收益有效区间会因 `T+1` 至 `T+6` 收益窗口自然提前结束。

## 三、从零全量运行

从零运行时不要添加 `--resume`。建议使用空的输出目录，避免目录中混入不属于
当前5,947因子配置的历史文件。程序不会主动删除输出根目录中的未知因子目录。

前台运行：

```bash
/data_all/duyj/miniconda3/envs/tmp/bin/python -u run.py \
  --pipeline \
  --full-factor-config config/factor_configs_5947.json \
  --start 2014-01-02 \
  --output-root outputs/zz500 \
  --workers 40
```

tmux后台运行：

```bash
mkdir -p logs

tmux new-session -d -s mlfactors_full \
'cd /data_all/duyj/MLFactors && exec \
  /data_all/duyj/miniconda3/envs/tmp/bin/python -u run.py \
    --pipeline \
    --full-factor-config config/factor_configs_5947.json \
    --start 2014-01-02 \
    --output-root outputs/zz500 \
    --workers 40 \
  > logs/run_full_pipeline.log 2>&1'
```

查看任务和日志：

```bash
tmux has-session -t mlfactors_full && echo "tmux任务存在"
tail -f /data_all/duyj/MLFactors/logs/run_full_pipeline.log
```

日志中的主要阶段标记为：

```text
BASE_VALUES_START / BASE_VALUES_DONE
BASE_MATRIX_START / BASE_MATRIX_DONE
DERIVED_VALUES_START / DERIVED_VALUES_DONE
BASE_MATRIX_CLEANED
FILL_START / FILL_DONE
REPORT_START / REPORT_DONE
```

任一并行任务失败时，日志会打印 `FAILED <factor>` 和异常堆栈；当前阶段结束后
主流程抛出异常并停止，不会在因子值阶段失败后继续生成报告。

### 中断后续跑

原命令增加 `--resume` 即可续跑：

```bash
/data_all/duyj/miniconda3/envs/tmp/bin/python -u run.py \
  --pipeline \
  --full-factor-config config/factor_configs_5947.json \
  --start 2014-01-02 \
  --output-root outputs/zz500 \
  --workers 40 \
  --resume
```

`--resume` 的检查口径是：因子目录至少存在一个CSV，且抽查的第一个CSV包含该
因子的 Raw 和 Neutralized 标准列。它是中断续跑的轻量检查，不会逐股票、逐日期
验证完整性。若怀疑某因子只写入了一部分文件，应删除或移走该因子输出目录后再
续跑，不能仅依赖 `--resume`。

续跑即使跳过已有因子值，仍会从199个基础因子输出重建派生计算所需的临时矩阵，
之后统一执行回填，并重新生成全部5,947份Markdown报告。

## 四、配置和派生公式

### 完整因子配置

`config/factor_configs_5947.json` 是正式全流程唯一的因子计算定义源。程序依据
`params.expansion_kind` 自动拆分：

- 不含 `expansion_kind`：199个基础因子；
- `single_transform`：2,388个单因子数学变换；
- `factor_combination`：3,360个多因子组合。

每个派生因子都记录：

- `formula`：公式定义；
- `input_factors`：输入基础因子；
- `input_directions`：输入因子的静态经济方向；
- `transformation` 或 `combination`：变换或组合编号；
- `description`，组合因子还可包含 `rule`。

单因子变换由 `transformation=T01...T12` 选择 `run.py` 中对应的通用数学实现，
`formula` 保存与该实现一致的公式。组合因子的 `formula` 由受限算术解析器直接
执行，只允许已声明基础因子的加、减、乘、除和括号，不执行任意Python代码。

因此，JSON已经完整保存全部派生因子的输入和公式定义，但仍需要 `run.py` 中的
通用执行器，以及 `factors/` 中199个基础因子的实现；JSON本身不是可独立运行的
程序。

### 组合公式中的 X、Z、S

对基础因子的每个版本分别构建以下矩阵：

- `X(factor)`：该版本的基础因子值；
- `Z(factor)`：逐日横截面稳健标准分（Robust Z-score）；
- `S(factor)`：`input_direction × Z(factor)`。

稳健标准分使用横截面中位数和MAD（Median Absolute Deviation，中位数绝对偏差），
尺度为 `1.4826 × MAD`；尺度无效时回退到标准差，最后裁剪到 `[-5, 5]`。

派生因子会分别从基础 Raw 矩阵和基础 Neutralized 矩阵计算两条路径。派生因子的
Neutralized结果在公式计算后，再按对数市值和申万一级行业做一次横截面中性化。

### 辅助配置

`config/factor_categories_5947.json` 是固定分类源，六类为动量、反转、量价、低波、
质量和成长。报告只读取已有分类，不在回测时重新推导。

`config/financial_factors_in_199.json` 标记21个直接依赖财报字段的基础因子。使用
这些基础因子的派生因子同样按财务因子处理，并输出距最近财报发布日期的天数。

`config/factor_configs_199.json` 只服务于“单独运行一个基础因子”的兼容入口，
不是正式全量流水线的依赖。

## 五、数据过滤、中性化和缺失值回填

数据加载后按每日中证500历史成分股过滤，并剔除当日ST和停牌股票。基础因子值
在过滤后的 `(date, symbol)` 样本上计算。

中性化（Neutralization）逐交易日执行横截面普通最小二乘回归
（Ordinary Least Squares, OLS）：

```text
factor = intercept + beta × log(market_cap) + SW一级行业哑变量 + residual
```

保存的中性化因子值是回归残差（Residual）。因子值非有限、市值非正、行业缺失，
或当日样本不足以完成回归时，对应中性化值保持为空。

时间窗口相关因子落盘后，对每只股票CSV中的 Raw 和 Neutralized 列分别执行：

```python
values.ffill().bfill()
```

即先用过去最近一个有效值向后填补中间空值（Forward fill），再用首个有效值向前
填补起始预热空值（Backward fill）。需要注意：

- 整列没有任何有效值时仍保持为空；
- 回填只修改CSV中已经存在的样本行；
- 因ST、停牌或不在当日成分股而被过滤掉的日期没有对应行，回填不会重新创建这些行。

## 六、正式输出和报告

正式全流程为每个因子生成：

```text
outputs/zz500/<factor>/
├── factors/
│   ├── 000006.SZ.csv
│   ├── ...
│   └── <symbol>.csv
└── factor_report.md
```

普通因子值CSV字段为：

```text
signal_date,available_date,<factor>_raw,<factor>_neutralization
```

基础财务因子及使用财务基础因子的派生因子还包含：

```text
days_since_latest_report_publish_date
```

正式报告阶段会删除旧流水线遗留的 `daily_factor_metrics.csv`、
`industry_daily_ic.csv`、各类summary CSV、HTML、PNG和 `yearly/` 目录，只保留
最终 `factor_report.md`。报告写入采用临时文件替换，避免正常写入过程中留下半份
Markdown。

每份报告由代码自动计算和生成，不需要大模型撰写结论。报告包含：

- 基本信息与空缺率；
- 核心Rank IC、ICIR、自然周IC胜率、失效周期、换手率和Top/Bottom 20%指标；
- IC衰减、五分组、Top/Bottom阈值、滞后检验；
- 市值、申万一级行业、市场状态和年度表现；
- Raw与Neutralized两个版本的独立结果。

字段定义和计算方法见
[`FACTOR_REPORT_FIELD_GUIDE.md`](FACTOR_REPORT_FIELD_GUIDE.md)。

## 七、单个基础因子兼容入口

调试基础因子时仍可单独运行：

```bash
/data_all/duyj/miniconda3/envs/tmp/bin/python run.py \
  --factor-config config/factor_configs_199.json \
  --factor BETA_5d \
  --start 2014-01-02 \
  --save-factor \
  --output-root outputs/zz500
```

该兼容入口会在单因子计算后立即评估，不经过正式全流程的“全部因子生成→统一
回填→统一报告”顺序，并可能生成日频或行业CSV、年度图片等兼容输出。因此它适合
开发调试，不应作为正式全量报告的生成方式。

派生因子不是注册表（Registry）中的普通基础因子，不能通过 `--factor` 单独运行，
必须使用 `--pipeline`。

## 八、可选宽表聚合

宽表聚合不是因子计算或报告依赖。只有其他模块需要“每只股票一张因子宽表”时，
才单独执行：

```bash
/data_all/duyj/miniconda3/envs/tmp/bin/python aggregate_factor_outputs.py \
  --outputs-dir outputs/zz500 \
  --output-dir outputs/factor_wide
```

也可以只聚合指定股票：

```bash
/data_all/duyj/miniconda3/envs/tmp/bin/python aggregate_factor_outputs.py \
  --outputs-dir outputs/zz500 \
  --output-dir outputs/factor_wide \
  --symbols 000006.SZ 000008.SZ
```

`aggregate_factor_outputs.py` 只读取标准的 Raw 和 Neutralized列，按日期横向合并，
并覆盖本次目标股票对应的宽表CSV。它不计算派生因子、不回填、不分类，也不生成
回测、仪表板、行业统计或报告。

## 九、不属于正式全流程的文件和目录

以下内容不会被 `run.py --pipeline` 读取：

| 文件或目录 | 当前定位 |
|---|---|
| `FACTOR_EXPANSION_PLAN_199.md` | 历史派生方案记录，可保留或删除，不影响从零运行 |
| `aggregate_factor_outputs.py` | 可选的独立宽表工具 |
| `tmp_analysis/` | 历史分析材料，不参与正式计算 |
| `tmp_data/` | 其他修复任务的临时数据目录，不参与中证500因子流水线 |
| `plot.py` | 仅供兼容或其他绘图流程使用，正式Markdown流水线不调用 |

`temporary_backtest_factor_expansions.py` 和 `generate_factor_categories.py` 已删除。
派生计算逻辑已经进入 `run.py`，因子类别已经固化在
`config/factor_categories_5947.json`。

## 十、核心模块职责

| 文件 | 职责 |
|---|---|
| `run.py` | 基础因子Runner、派生公式执行、回填和5,947因子正式流水线 |
| `dataloader.py` | 行情读取、证券代码归一、历史成分股及ST/停牌过滤 |
| `factors/` | 199个基础因子实现和注册表 |
| `neutralization.py` | 对数市值＋申万一级行业OLS残差中性化 |
| `factors_eval.py` | Rank IC、ICIR、收益分组、换手率和回撤等指标计算 |
| `factor_report.py` | 公式展示、类别读取和最终Markdown报告排版 |
| `aggregate_factor_outputs.py` | 可选的纯宽表聚合工具，不参与正式全流程 |

## 十一、代码验证

推荐工作目录仍为 `/data_all/duyj/MLFactors`。

运行当前单元测试：

```bash
/data_all/duyj/miniconda3/envs/tmp/bin/python -m unittest discover \
  -s tests -p 'test_*.py'
```

检查正式入口参数：

```bash
/data_all/duyj/miniconda3/envs/tmp/bin/python run.py --help
```

这些测试覆盖因子分类完整性、派生配置自包含性、自然周IC胜率、IC衰减和主要
评估字段结构；不等同于重新运行5,947个因子的全量数值回归测试。
