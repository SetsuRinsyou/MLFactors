# 232因子滚动机器学习组合

数据源固定为 `factor_results_232/zz500_factor_wide`。`convert_to_qlib.py`
将232个因子、行情和未来5日标签转换到 `data/zz500/dataset.parquet`，原始
CSV保持只读，因子缺失值只在模型运行内存中填0。

三种模型使用同一滚动规则。对预测月 `M`：

- 训练集从首个因子覆盖率大于30%的交易日起，截止 `M-13` 月月末。
- 测试集为 `M-12` 月月初至 `M-1` 月月末，固定12个月。
- 训练集和测试集各自清除最后6个信号日，保证
  `adj_close[t+6] / adj_close[t+1] - 1` 不越过区间边界。
- 每个窗口从头训练10轮，以测试集 Top 20% 阶段收益率最高的一轮预测 `M`。
- 阶段收益率不是年化值：将每日Top 20%股票等权组成5日组合，按5种交易日
  偏移形成互不重叠的复利路径，再等权平均五条路径的期末总收益。

每个模型目录只保留 `config.yaml`、训练代码及运行产物：

```text
MODEL/
├── log/epochs.log
├── model/latest.{joblib|txt|pt}
└── output/
    ├── factors/
    │   ├── 000001.SZ.csv
    │   └── ...
    └── backtest/
        ├── factor_summary.csv
        ├── yearly_summary.csv
        └── daily_factor_metrics.csv
```

因子文件仅覆盖2020年1月至数据集最后日期，每只股票一份CSV，列为
`signal_date, available_date, MODEL`。`available_date`取全市场交易日历中的
下一交易日；当前数据集最后信号日2026-07-22的下一交易日在配置中显式设为
2026-07-23。

标签仍为每日未来5日超额收益前30%记 `+1`、后30%记 `-1`，中间40%不进入
损失。预处理仍为逐日5倍MAD去极值、截面Z-score和运行时缺失值填0。
SVM使用Hinge Loss，LightGBM使用Binary Log Loss，MLP使用标签平滑
BCEWithLogits Loss；三者每轮均记录训练集与测试集的loss、IC、ICIR、
Top 20%阶段收益率和Top 20%夏普比率。

## 回测

`backtest.py` 读取任意 `ml/METHOD/output/factors`，使用中证500历史成分股及
停牌/ST过滤规则重新加载原始行情。真实5日收益严格按
`adj_close[t+6] / adj_close[t+1] - 1` 计算。总时段为2020-01-01至因子最后
日期；同时对2020至2026每个自然年独立切片并重新计算，年度末不会借用下一年
价格。回测不划分大盘状态。

除原有IC、ICIR、TimingIC、分层累计收益、夏普比率、最大回撤和胜率等指标外，
新增两个日频双向排名效率，并在总时段与年度汇总中取时间序列均值。每日有效
股票数为 `n`，选股数为 `k = ceil(20% × n)`，理想排名和为
`k(k+1)/2`：

- `top_20%_selection_rank_efficiency`：理想排名和除以“因子Top 20%股票
  在真实收益降序排名中的名次和”，中文名为Top 20%选股排名效率。
- `top_20%_return_capture_rank_efficiency`：理想排名和除以“真实收益
  Top 20%股票在因子降序排名中的名次和”，中文名为Top 20%收益捕获排名效率。

两项指标均位于 `(0, 1]`，越大越好；完全匹配时为1，随机排序的理论基准约为
0.2。当 `k=20` 时，理想排名和就是 `1+2+...+20=210`。

推荐工作目录：`/data_all/dyj/MLFactors`

```bash
/data_all/dyj/miniconda3/envs/fshc/bin/python ml/convert_to_qlib.py
/data_all/dyj/miniconda3/envs/fshc/bin/python ml/SVM/train.py
/data_all/dyj/miniconda3/envs/fshc/bin/python ml/LightGBM/train.py
/data_all/dyj/miniconda3/envs/fshc/bin/python ml/MLP/train.py --gpu-id 0
```

分别回测三种方法：

```bash
/data_all/dyj/miniconda3/envs/fshc/bin/python ml/backtest.py SVM
/data_all/dyj/miniconda3/envs/fshc/bin/python ml/backtest.py LightGBM
/data_all/dyj/miniconda3/envs/fshc/bin/python ml/backtest.py MLP
```

新增方法只需满足目录为 `ml/METHOD/output/factors`、因子值列名为 `METHOD`，
即可直接运行 `python ml/backtest.py METHOD`。不同目录或列名可通过
`--factor-dir`、`--output-dir` 和 `--factor-column` 显式指定。
