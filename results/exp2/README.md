# 实验2: 不同持仓周期的因子衰减分析

## 实验目标
分析因子在1日/5日/10日/20日四个持仓周期的表现衰减情况，对比Baseline（单体LLM）和本研究方法（多智能体辩论）的效果。

## 数据划分
- **训练期**: 2017-01-03 至 2022-06-26（用于因子挖掘）
- **测试期**: 2022-06-27 至 2023-06-26（仅用于回测验证）

## 运行方式
```bash
python run_experiment2.py
```

## 输出文件
- `results/exp2/exp2_results.json` - 详细结果（JSON格式）
- `results/exp2/exp2_summary.csv` - 汇总表格（CSV格式）
- `results/exp2/exp2_aggregated.json` - 统计汇总（JSON格式）

## 核心指标说明

### 1. Rank IC
因子值排名与未来收益排名的Spearman相关系数。

### 2. Raw IR
年化超额收益 / 年化跟踪误差（未扣成本）。

### 3. 净IR
(年化超额收益 - 0.15% × 日均换手率) / 年化跟踪误差。

### 4. 日均换手率
基于因子值自相关计算：turnover = 1 - autocorr。

### 5. IC半衰期
Rank IC衰减至1日周期IC的50%所需的周期数（插值估算）。

### 6. 信息留存率
各周期IC / 1日周期IC × 100%。

## 策略构建
- 每期根据因子值分五档
- 做多前20%（Top分位数），做空后20%（Bottom分位数）
- 组合内部等权配置
- 策略收益 = 多头组合收益率 - 空头组合收益率

## 对比设置
| 维度 | Baseline组 | 本方法组 |
|------|-----------|---------|
| 优化方式 | Refiner Agent单智能体 | 多智能体辩论(Agent_A + Agent_B + Critic) |
| 挖掘数量 | 每种类型5个，共15个 | 每种类型5个，共15个 |
| 因子类型 | 动量/价值/波动率 | 动量/价值/波动率 |

## 代码改造要点

### 1. factor_calculator.py
```python
# 新增holding_period参数
class FactorCalculator:
    def __init__(self, ..., holding_period: int = 5):
        ...
        self.holding_period = holding_period
```

### 2. evaluator.py
新增三个函数：
- `evaluate_by_holding_period()` - 按持仓周期评估
- `evaluate_all_periods()` - 评估四个周期
- `calculate_portfolio_turnover()` - 计算组合换手率

### 3. run_experiment2.py
主要组件：
- `BaselineMCTS` - 使用Refiner Agent的MCTS变体
- `run_single_experiment()` - 单组实验运行
- `evaluate_in_test_period()` - 测试期评估
- `print_key_conclusions()` - 打印关键结论

## 预期结果参考
根据实验设计，预期结果：
- 1日周期本方法Rank IC ~0.052，Baseline ~0.038
- 本方法半衰期 ~12日，Baseline ~7日
- 5日周期净IR最优：本方法 ~0.78，Baseline ~0.52
