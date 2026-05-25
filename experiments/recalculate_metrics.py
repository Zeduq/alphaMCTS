import json
import pandas as pd
import os
import sys
from pathlib import Path

# 确保能正确引入项目的各个模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from search.tree import AlphaFormula
# 直接导入已经加载好行情的计算器和评估函数
from evaluation.evaluator import calculator, get_alphalens_metrics


def process_and_recalculate(json_file, output_csv):
    print(f"========== 开始处理文件: {json_file} ==========")
    if not os.path.exists(json_file):
        print(f"⚠️ 找不到文件: {json_file}，请检查文件名或路径。")
        return

    with open(json_file, 'r', encoding='utf-8') as f:
        factors = json.load(f)

    results = []
    for factor in factors:
        name = factor.get("Name", "Unknown")
        print(f"-> 正在回测重算因子: {name} ...")

        steps = factor.get("Formula_Steps")
        formula_str = ""

        # 针对不同的 JSON 结构采取不同的处理方式
        if isinstance(steps, str):
            # debate_result_converted.json 里的是纯字符串公式
            formula_str = steps
        elif isinstance(steps, list):
            # elite_factors_refiner 里的是字典列表（缺时间窗口参数），我们需要帮它补齐
            args_dict = {}
            for step in steps:
                for p in step.get("param", []):
                    if p not in args_dict:
                        p_lower = p.lower()
                        # 根据参数名字赋予常见的量化窗口整数
                        if "short" in p_lower:
                            args_dict[p] = 5
                        elif "long" in p_lower:
                            args_dict[p] = 20
                        elif "decay" in p_lower:
                            args_dict[p] = 10
                        else:
                            args_dict[p] = 20  # 默认赋予月频 20 天窗口

            try:
                # 重新将补齐参数的结构化数据实例化，转化为底层可解析的计算表达式
                formula_obj = AlphaFormula(
                    name=name,
                    description="",
                    formula_steps=steps,
                    arguments=[args_dict]
                )
                formula_str = formula_obj.to_expression_string()
            except Exception as e:
                print(f"  ❌ 公式解析失败，跳过该因子: {e}")
                continue
        else:
            print("  ❌ 未知公式格式，跳过。")
            continue

        # 将生成的最终公式字符串丢进底层 Pandas / NumPy 计算器
        factor_values = None
        metrics = {}
        try:
            factor_values = calculator.calculate_factor(formula_str)
            if factor_values is not None:
                if isinstance(factor_values, pd.DataFrame):
                    factor_values = factor_values.stack()

                # 直接调用刚刚升级过、能算普通 Pearson IC 的回测指标函数
                metrics = get_alphalens_metrics(factor_values)
                if not metrics:
                    metrics = {}
        except Exception as e:
            print(f"  ❌ 因子计算异常，回测失败: {e}")
            pass

        # 无论成功与否，只提取你所要求的四个金融指标
        results.append({
            "Name": name,
            "IC": round(metrics.get("ic_mean", 0.0), 4),
            "Rank IC": round(metrics.get("rank_ic_mean", 0.0), 4),
            "AR": round(metrics.get("annualized_return", 0.0), 4),
            "IR": round(metrics.get("information_ratio", 0.0), 4)
        })

    # 导出为扁平化的 CSV 表格
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 文件 {json_file} 处理完毕！")
    print(f"📊 重新计算的新指标已导出至: {output_csv}\n")


if __name__ == "__main__":
    # 配置要处理的输入 JSON 文件和输出的 CSV 文件名
    files_to_process = [
        ("debate_result_converted.json", "recalculated_metrics_debate.csv"),
        ("elite_factors_refiner_20260320_1844.json", "recalculated_metrics_refiner.csv")
    ]

    for json_path, csv_path in files_to_process:
        process_and_recalculate(json_path, csv_path)
