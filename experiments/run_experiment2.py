import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

from config import (
    PROMPT_DIR, DATA_DIR, 
    TRAIN_BEGIN, TRAIN_END, TEST_BEGIN, TEST_END,
    ELITE_Q_THRESHOLD
)
from search.lifecycle import MCTS
from search.tree import AlphaNode, AlphaFormula
from utils.exporter import export_elite_factors
from constraint.library import AlphaLibrary
from inference.agents.portrait import PortraitAgent
from inference.agents.formula import FormulaAgent
from inference.agents.refiner import RefinerAgent
from evaluation.evaluator import simulate_evaluation, evaluate_all_periods


# 实验配置 - 从 config.py 导入
HOLDING_PERIODS = [1, 5, 10, 20]
FACTOR_TYPES = ["动量因子", "价值因子", "波动率因子"]
TARGET_COUNT_PER_TYPE = 5  # 每种类型5个因子
TOTAL_TARGET = 15  # 总共15个因子

RESULTS_DIR = "results/exp2"


class BaselineMCTS(MCTS):
    
    def __init__(self, root: AlphaNode):
        super().__init__(root)
        self.refiner_agent = RefinerAgent(
            prompt_path=os.path.join(PROMPT_DIR, "alpha_refinement.txt")
        )
    
    def expand(self, node_to_expand: AlphaNode, freq_subtrees: List[str], 
               alpha_repo: AlphaLibrary, **kwargs) -> Any:
        from evaluation.evaluator import get_refinement_dimension
        
        print(f"\n--- [Baseline] 正在扩展节点: {node_to_expand.portrait.get('name', '未命名')} ---")
        
        refinement_dim = get_refinement_dimension(node_to_expand.scores)
        print(f"[Baseline] 优化目标维度: {refinement_dim}")
        
        current_factor_type = node_to_expand.portrait.get('factor_type', '综合型')
        
        # 构建简单的优化建议
        suggestions = f"请优化因子的{refinement_dim}维度，保持{current_factor_type}类型特征。"
        if freq_subtrees:
            suggestions += f" 避免使用以下常见结构: {freq_subtrees}"
        
        # 调用Refiner Agent（单智能体）
        print("[Baseline] 调用Refiner Agent进行优化...")
        new_portrait = self.refiner_agent.execute(
            original_formula=node_to_expand.formula,
            original_portrait=node_to_expand.portrait,
            suggestions=suggestions,
            freq_subtrees=freq_subtrees
        )
        
        if not new_portrait:
            print("[Baseline] Refiner未能生成新画像，跳过本次扩展。")
            return None
        
        new_portrait['factor_type'] = current_factor_type
        
        # 生成公式
        print("[Baseline] 正在生成新公式...")
        formula_agent = FormulaAgent(prompt_path=os.path.join(PROMPT_DIR, "formula_generation.txt"))
        new_formula = formula_agent.execute(alpha_portrait=new_portrait)
        
        if not new_formula:
            print("[Baseline] 公式生成失败，跳过。")
            return None
        
        # 创建新节点
        new_node = AlphaNode(
            formula=new_formula,
            portrait=new_portrait,
            parent=node_to_expand,
            refinement_summary=f"[Baseline]经Refiner优化(目标:{refinement_dim})"
        )
        
        # 评估
        print("[Baseline] 正在评估新因子...")
        new_scores = simulate_evaluation(new_formula, new_node, alpha_repo)
        new_node.scores = new_scores
        new_node.q_value = np.mean(list(new_scores.values())) if new_scores else 0.0
        
        node_to_expand.children.append(new_node)
        print(f"[Baseline] 新节点创建完成: Q值={new_node.q_value:.2f}")
        
        return new_node


def initialize_root_node(factor_type: str) -> AlphaNode:
    print(f"\n{'='*60}")
    print(f"正在初始化 [{factor_type}] 的根节点")
    print(f"{'='*60}")
    
    portrait_agent = PortraitAgent(prompt_path=os.path.join(PROMPT_DIR, "portrait_generation.txt"))
    formula_agent = FormulaAgent(prompt_path=os.path.join(PROMPT_DIR, "formula_generation.txt"))
    
    # 生成初始画像
    root_portrait = portrait_agent.execute(freq_subtrees=[], factor_type=factor_type)
    if not root_portrait:
        raise Exception("生成初始Alpha画像失败。")
    
    # 生成初始公式
    root_formula = formula_agent.execute(alpha_portrait=root_portrait)
    if not root_formula:
        raise Exception("合成初始Alpha公式失败。")
    
    if isinstance(root_formula, dict):
        root_formula = AlphaFormula(**root_formula)
    
    root_node = AlphaNode(formula=root_formula, portrait=root_portrait)
    
    # 评估根节点
    print("正在评估初始根节点...")
    root_scores = simulate_evaluation(root_node.formula, root_node, AlphaLibrary())
    root_node.scores = root_scores
    root_node.q_value = np.mean(list(root_scores.values())) if root_scores else 0.0
    
    print(f"✅ 根节点初始化完成! 初始Q值: {root_node.q_value:.2f}")
    return root_node


def run_single_experiment(method: str, factor_type: str, target_count: int = 5) -> List[AlphaNode]:
    print(f"\n{'#'*70}")
    print(f"# 开始实验: Method={method.upper()}, Type={factor_type}")
    print(f"{'#'*70}")
    
    alpha_repo = AlphaLibrary()
    
    # 初始化根节点
    try:
        root_node = initialize_root_node(factor_type)
    except Exception as e:
        print(f"初始化失败: {e}")
        return []
    
    # 创建MCTS实例
    if method == "baseline":
        mcts = BaselineMCTS(root=root_node)
    else:
        mcts = MCTS(root=root_node)
    
    elite_pool = []
    budget = 100  # 搜索预算
    
    # 如果根节点质量足够，先入库
    if root_node.q_value > 5.0:
        elite_pool.append(root_node)
        print(f"🎉 根节点直接入库! 当前进度: {len(elite_pool)}/{target_count}")
    
    # 主搜索循环
    while len(elite_pool) < target_count and budget > 0:
        print(f"\n[Budget: {budget} | 已入库: {len(elite_pool)}/{target_count}]")
        
        # 挖掘频繁子树
        from constraint.fsa import mine_frequent_subtrees
        freq_subtrees = mine_frequent_subtrees(alpha_repo.alphas, top_k=3)
        
        # 选择节点
        node_to_expand = mcts.select()
        
        # 扩展节点
        if method == "baseline":
            new_node = mcts.expand(node_to_expand, freq_subtrees, alpha_repo)
        else:
            new_node = mcts.expand(node_to_expand, freq_subtrees, alpha_repo)
        
        if new_node:
            # 反向传播
            mcts.backpropagate(new_node)
            
            # 判断入库（Q值>5.0）
            if new_node.q_value > ELITE_Q_THRESHOLD:
                elite_pool.append(new_node)
                metrics = getattr(new_node, 'financial_metrics', {})
                print(f"🎉 因子入库! [{len(elite_pool)}/{target_count}] "
                      f"Rank IC: {metrics.get('rank_ic_mean', 0):.4f}, Q: {new_node.q_value:.2f}")
        
        budget -= 1
    
    print(f"\n{'='*60}")
    print(f"实验结束: 共捕获 {len(elite_pool)} 个因子")
    print(f"{'='*60}")
    
    return elite_pool


def evaluate_in_test_period(factors: List[AlphaNode], method: str, factor_type: str) -> List[Dict]:
    results = []
    
    for i, node in enumerate(factors):
        print(f"\n评估 [{method}/{factor_type}] 第 {i+1}/{len(factors)} 个因子...")
        
        # 在测试期评估四个周期
        period_metrics = evaluate_all_periods(node.formula)
        
        result = {
            "method": method,
            "factor_type": factor_type,
            "factor_name": node.portrait.get('name', f'Unknown_{i}'),
            "factor_formula": node.formula.to_expression_string(),
            "train_q_value": node.q_value,
            "period_metrics": {}
        }
        
        # 整理各周期指标
        for period in HOLDING_PERIODS:
            if period in period_metrics:
                m = period_metrics[period]
                result["period_metrics"][period] = {
                    "rank_ic": round(m.get("rank_ic", 0), 4),
                    "rank_ic_std": round(m.get("rank_ic_std", 0), 4),
                    "icir": round(m.get("icir", 0), 4),
                    "raw_ir": round(m.get("raw_ir", 0), 4),
                    "net_ir": round(m.get("net_ir", 0), 4),
                    "daily_turnover": round(m.get("daily_turnover", 0), 4),
                    "portfolio_turnover": round(m.get("portfolio_turnover", 0), 4),
                    "annual_return": round(m.get("annual_return", 0), 4),
                    "tracking_error": round(m.get("tracking_error", 0), 4),
                    "info_retention": round(m.get("info_retention", 0), 2),
                    "ic_half_life": round(m.get("ic_half_life", 0), 2)
                }
        
        results.append(result)
    
    return results


def aggregate_results(all_results: List[Dict]) -> Dict:
    summary = defaultdict(lambda: defaultdict(list))
    
    for result in all_results:
        method = result["method"]
        period_metrics = result["period_metrics"]
        
        for period in HOLDING_PERIODS:
            if period in period_metrics:
                m = period_metrics[period]
                for metric in ["rank_ic", "raw_ir", "net_ir", "ic_half_life", "info_retention"]:
                    if metric in m:
                        summary[method][f"{metric}_{period}d"].append(m[metric])
    
    # 计算均值
    aggregated = {}
    for method in ["baseline", "debate"]:
        aggregated[method] = {}
        for key, values in summary[method].items():
            aggregated[method][key] = round(np.mean(values), 4) if values else 0.0
    
    return aggregated


def generate_summary_csv(all_results: List[Dict], output_path: str):
    rows = []
    
    for result in all_results:
        row_base = {
            "method": result["method"],
            "factor_type": result["factor_type"],
            "factor_name": result["factor_name"],
            "train_q_value": result["train_q_value"]
        }
        
        for period in HOLDING_PERIODS:
            if period in result["period_metrics"]:
                m = result["period_metrics"][period]
                row = {
                    **row_base,
                    "holding_period": f"{period}D",
                    "rank_ic": m.get("rank_ic", 0),
                    "icir": m.get("icir", 0),
                    "raw_ir": m.get("raw_ir", 0),
                    "net_ir": m.get("net_ir", 0),
                    "daily_turnover": m.get("daily_turnover", 0),
                    "portfolio_turnover": m.get("portfolio_turnover", 0),
                    "annual_return": m.get("annual_return", 0),
                    "tracking_error": m.get("tracking_error", 0),
                    "info_retention": m.get("info_retention", 0),
                    "ic_half_life": m.get("ic_half_life", 0)
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ CSV结果已保存: {output_path}")
    return df


def print_key_conclusions(aggregated: Dict):
    print("\n" + "="*70)
    print("                    实验2 关键结论")
    print("="*70)
    
    # 1日周期Rank IC对比
    baseline_ic_1d = aggregated.get("baseline", {}).get("rank_ic_1d", 0)
    debate_ic_1d = aggregated.get("debate", {}).get("rank_ic_1d", 0)
    print(f"\n1. Rank IC对比 (1日周期):")
    print(f"   - Baseline: {baseline_ic_1d:.4f}")
    print(f"   - 本方法:   {debate_ic_1d:.4f}")
    print(f"   - 提升:     {(debate_ic_1d - baseline_ic_1d):.4f} ({(debate_ic_1d/baseline_ic_1d - 1)*100:.1f}%)")
    
    # IC半衰期对比
    baseline_halflife = aggregated.get("baseline", {}).get("ic_half_life_1d", 0)
    debate_halflife = aggregated.get("debate", {}).get("ic_half_life_1d", 0)
    print(f"\n2. IC半衰期对比:")
    print(f"   - Baseline: {baseline_halflife:.1f}日")
    print(f"   - 本方法:   {debate_halflife:.1f}日")
    print(f"   - 提升:     {(debate_halflife - baseline_halflife):.1f}日")
    
    # 各周期净IR对比
    print(f"\n3. 净IR对比 (各持仓周期):")
    print(f"   {'周期':<8} {'Baseline':<12} {'本方法':<12} {'提升':<12}")
    print(f"   {'-'*44}")
    best_period = None
    best_improvement = -999
    for period in HOLDING_PERIODS:
        baseline_ir = aggregated.get("baseline", {}).get(f"net_ir_{period}d", 0)
        debate_ir = aggregated.get("debate", {}).get(f"net_ir_{period}d", 0)
        improvement = debate_ir - baseline_ir
        print(f"   {period}D{chr(12288)*4} {baseline_ir:<12.4f} {debate_ir:<12.4f} {improvement:<+12.4f}")
        if improvement > best_improvement:
            best_improvement = improvement
            best_period = period
    
    print(f"\n4. 最优持仓周期: {best_period}日 (净IR提升最大)")
    
    # 信息留存率
    print(f"\n5. 信息留存率 (各周期IC / 1日IC):")
    print(f"   {'周期':<8} {'Baseline':<12} {'本方法':<12}")
    print(f"   {'-'*32}")
    for period in [5, 10, 20]:
        baseline_ret = aggregated.get("baseline", {}).get(f"info_retention_{period}d", 0)
        debate_ret = aggregated.get("debate", {}).get(f"info_retention_{period}d", 0)
        print(f"   {period}D{chr(12288)*4} {baseline_ret:<12.1f}% {debate_ret:<12.1f}%")
    
    print("\n" + "="*70)


def main():
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "        实验2: 不同持仓周期的因子衰减分析".center(56) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    # 创建结果目录
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    all_results = []
    
    # 遍历每种因子类型
    for factor_type in FACTOR_TYPES:
        # 运行Baseline实验
        baseline_factors = run_single_experiment("baseline", factor_type, TARGET_COUNT_PER_TYPE)
        baseline_results = evaluate_in_test_period(baseline_factors, "baseline", factor_type)
        all_results.extend(baseline_results)
        
        # 运行Debate实验
        debate_factors = run_single_experiment("debate", factor_type, TARGET_COUNT_PER_TYPE)
        debate_results = evaluate_in_test_period(debate_factors, "debate", factor_type)
        all_results.extend(debate_results)
    
    # 保存详细结果（JSON）
    json_path = os.path.join(RESULTS_DIR, "exp2_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 详细结果已保存: {json_path}")
    
    # 生成CSV汇总
    csv_path = os.path.join(RESULTS_DIR, "exp2_summary.csv")
    df = generate_summary_csv(all_results, csv_path)
    
    # 汇总统计
    aggregated = aggregate_results(all_results)
    
    # 保存汇总结果
    summary_json_path = os.path.join(RESULTS_DIR, "exp2_aggregated.json")
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)
    print(f"✅ 汇总结果已保存: {summary_json_path}")
    
    # 打印关键结论
    print_key_conclusions(aggregated)
    
    print("\n" + "#"*70)
    print("#                         实验2 完成")
    print("#"*70)


if __name__ == "__main__":
    main()

