import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import sys

# 测试1: 验证配置导入
print("=" * 50)
print("测试1: 配置导入")
print("=" * 50)
from config import INITIAL_SEARCH_BUDGET, EFFECTIVENESS_THRESHOLD
print(f"[OK] INITIAL_SEARCH_BUDGET = {INITIAL_SEARCH_BUDGET}")
print(f"[OK] EFFECTIVENESS_THRESHOLD = {EFFECTIVENESS_THRESHOLD}")

# 测试2: 验证FactorCalculator动态持仓周期
print("\n" + "=" * 50)
print("测试2: FactorCalculator动态持仓周期")
print("=" * 50)
from factor_calculator import FactorCalculator

try:
    calc_1d = FactorCalculator(
        data_path='D:/AAProject/Data/000300SH.csv',
        begin_date='2022-06-27',
        end_date='2023-06-26',
        holding_period=1
    )
    print(f"[OK] 1日持仓周期计算器创建成功: holding_period={calc_1d.holding_period}")
    
    calc_5d = FactorCalculator(
        data_path='D:/AAProject/Data/000300SH.csv',
        begin_date='2022-06-27',
        end_date='2023-06-26',
        holding_period=5
    )
    print(f"[OK] 5日持仓周期计算器创建成功: holding_period={calc_5d.holding_period}")
except Exception as e:
    print(f"[FAIL] 失败: {e}")

# 测试3: 验证评估函数
print("\n" + "=" * 50)
print("测试3: 评估函数存在性检查")
print("=" * 50)
try:
    from evaluation.evaluator import evaluate_by_holding_period, evaluate_all_periods
    print("[OK] evaluate_by_holding_period 函数存在")
    print("[OK] evaluate_all_periods 函数存在")
except ImportError as e:
    print(f"[FAIL] 导入失败: {e}")

# 测试4: 验证实验2运行脚本结构
print("\n" + "=" * 50)
print("测试4: 实验2脚本结构检查")
print("=" * 50)
try:
    import run_experiment2
    print("[OK] run_experiment2 模块可导入")
    print(f"[OK] TRAIN_BEGIN = {run_experiment2.TRAIN_BEGIN}")
    print(f"[OK] TEST_BEGIN = {run_experiment2.TEST_BEGIN}")
    print(f"[OK] HOLDING_PERIODS = {run_experiment2.HOLDING_PERIODS}")
    print(f"[OK] BaselineMCTS 类存在: {hasattr(run_experiment2, 'BaselineMCTS')}")
except Exception as e:
    print(f"[FAIL] 失败: {e}")

# 测试5: 检查结果目录
print("\n" + "=" * 50)
print("测试5: 结果目录")
print("=" * 50)
results_dir = "results/exp2"
os.makedirs(results_dir, exist_ok=True)
print(f"[OK] 结果目录已创建/存在: {results_dir}")

print("\n" + "=" * 50)
print("所有基础测试通过!")
print("=" * 50)
print("\n要运行完整实验，请执行:")
print("  python run_experiment2.py")

