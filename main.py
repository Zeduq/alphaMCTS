import json
import os
import numpy as np
import traceback
from datetime import datetime
from config import PROMPT_DIR
from mcts.search import MCTS
from utils.data_structures import AlphaNode, AlphaFormula
from agents.portrait_agent import PortraitAgent
from agents.formula_agent import FormulaAgent
from evaluation.evaluator import simulate_evaluation
from config import INITIAL_SEARCH_BUDGET, BUDGET_INCREMENT, EFFECTIVENESS_THRESHOLD
from alpha_library.library import AlphaLibrary
from fsa.fsa_miner import mine_frequent_subtrees


# (initialize_root_node 函数保持不变)
def initialize_root_node(factor_type: str) -> AlphaNode:
    print(f"--- 正在初始化根节点 (使用提示词目录: {PROMPT_DIR}) ---")
    # 使用 os.path.join 拼接路径
    portrait_agent = PortraitAgent(prompt_path=os.path.join(PROMPT_DIR, "portrait_generation.txt"))
    formula_agent = FormulaAgent(prompt_path=os.path.join(PROMPT_DIR, "formula_generation.txt"))
    root_portrait = portrait_agent.execute(freq_subtrees=[], factor_type=factor_type)
    if not root_portrait: raise Exception("生成初始Alpha画像失败。")
    root_formula = formula_agent.execute(alpha_portrait=root_portrait)
    if not root_formula: raise Exception("合成初始Alpha公式失败。")
    root_node = AlphaNode(formula=root_formula, portrait=root_portrait)
    # 评估根节点时，传入一个空的AlphaLibrary实例
    root_scores = simulate_evaluation(root_node.formula, root_node, AlphaLibrary())
    root_node.scores = root_scores
    root_node.q_value = np.mean(list(root_scores.values())) if root_scores else 0
    print("--- 根节点详细信息 ---")
    print(f"根节点 '{root_node.portrait.get('name', '未命名')}' 已创建, Q值为: {root_node.q_value:.2f}")
    formula_str = root_node.formula.to_expression_string()
    print(f"  因子公式: {formula_str}")
    scores_str = json.dumps(root_node.scores, indent=4)
    print(f"  五维得分:\n{scores_str}")
    # [新增] 打印根节点的金融指标
    metrics_str = json.dumps(root_node.financial_metrics, indent=4, default=lambda x: f"{x:.4f}")
    print(f"  金融指标:\n{metrics_str}")
    print("--------------------")
    return root_node


def run_search(factor_type: str):
    """
    运行MCTS搜索过程的主函数。
    """
    all_generated_nodes_data = []
    try:
        root_node = initialize_root_node(factor_type)
        is_root_effective = root_node.scores.get("Effectiveness", 0) >= EFFECTIVENESS_THRESHOLD
        all_generated_nodes_data.append({
            "node": root_node, "in_library": is_root_effective
        })
    except Exception as e:
        print("\n--- 初始化过程中发生错误 ---")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        print("--- 错误追踪 ---")
        traceback.print_exc()
        print("-----------------\n")
        return

    mcts = MCTS(root=root_node)
    effective_alpha_repository = AlphaLibrary()
    if is_root_effective:
        effective_alpha_repository.add(root_node)
    max_score_overall = root_node.q_value
    search_budget = INITIAL_SEARCH_BUDGET
    i = 0
    while i < search_budget:
        print(f"\n{'=' * 20} MCTS 迭代: {i + 1}/{search_budget} {'=' * 20}")
        frequent_subtrees = mine_frequent_subtrees(effective_alpha_repository.alphas, top_k=3)
        node_to_expand = mcts.select()
        new_node = mcts.expand(node_to_expand, freq_subtrees=frequent_subtrees, alpha_repo=effective_alpha_repository)
        if new_node:
            mcts.backpropagate(new_node)
            is_new_node_effective = new_node.scores.get("Effectiveness", 0) >= EFFECTIVENESS_THRESHOLD
            all_generated_nodes_data.append({
                "node": new_node, "in_library": is_new_node_effective
            })
            if new_node.q_value > max_score_overall:
                max_score_overall = new_node.q_value
                search_budget += BUDGET_INCREMENT
                print(f"*** 发现新的最佳Alpha, Q值: {max_score_overall:.2f}。搜索预算增加至: {search_budget} ***")
            if is_new_node_effective:
                effective_alpha_repository.add(new_node)
        i += 1

    print("\n\n--- MCTS 搜索完成 ---")
    print(f"总共生成因子数: {len(all_generated_nodes_data)}")
    print(f"仓库中有效Alpha总数: {len(effective_alpha_repository)}")

    # [修改] 写入 result.txt 的逻辑
    with open("result.txt", "w", encoding="utf-8") as f:
        run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"==================== RUN START: {run_timestamp} ====================\n")
        f.write(f"Factor Type: {factor_type}\n")
        if not all_generated_nodes_data:
            f.write("No alphas were generated in this run.\n")
        for idx, data in enumerate(all_generated_nodes_data):
            node = data["node"];
            in_library = data["in_library"]
            portrait = node.portrait;
            formula_obj = node.formula;
            scores = node.scores;
            q_value = node.q_value
            financial_metrics = node.financial_metrics  # <-- 获取指标

            formula_str_val = formula_obj.to_expression_string() if isinstance(formula_obj, AlphaFormula) else "公式解析失败"
            scores_json_str = json.dumps(scores)
            metrics_json_str = json.dumps(financial_metrics, default=lambda x: f"{x:.4f}")  # <-- 序列化指标

            f.write(f"--- Alpha {idx + 1} ---\n")
            f.write(f"Name: {portrait.get('name', '未命名Alpha')}\n")
            f.write(f"Q-Value: {q_value:.4f}\n")
            f.write(f"In Library: {'Yes' if in_library else 'No'}\n")
            f.write(f"Formula: {formula_str_val}\n")
            f.write(f"Scores: {scores_json_str}\n")
            f.write(f"Financial Metrics: {metrics_json_str}\n")  # <-- 写入文件

        f.write(f"==================== RUN END ====================\n\n")

    print("\n所有生成因子的结果已追加到 result.txt 文件中。")

    # [修改] 最终打印到控制台的逻辑
    best_alphas = effective_alpha_repository.get_best_alphas(n=10)
    print(f"\n--- 仓库中排名前 {len(best_alphas)} 的Alpha ---")
    for idx, alpha_data in enumerate(best_alphas):
        portrait = alpha_data.get('portrait', {})
        formula_obj = alpha_data.get('formula')
        scores = alpha_data.get('scores', {})
        q_value = alpha_data.get('q_value', 0)
        financial_metrics = alpha_data.get('financial_metrics', {})  # <-- 获取指标

        name_str = f"{idx + 1}. 名称: {portrait.get('name', '未命名Alpha')}"
        q_str = f"   Q值: {q_value:.4f}"
        desc_str = f"   描述: {portrait.get('description', '无描述')}"
        formula_str_val = formula_obj.to_expression_string() if isinstance(formula_obj, AlphaFormula) else "公式解析失败"
        formula_str = f"   公式: {formula_str_val}"
        scores_pretty_str = json.dumps(scores, indent=4)
        scores_str = f"   五维得分:\n{scores_pretty_str}"

        print(name_str)
        print(q_str)
        print(desc_str)
        print(formula_str)
        print(scores_str)

        # <-- [新增] 格式化打印金融指标
        print("   金融指标:")
        print(f"     IC/IR (icir): {financial_metrics.get('icir', 0.0):.4f}")
        print(f"     年化收益率: {financial_metrics.get('annualized_return', 0.0):.4f}")
        print(f"     夏普比率: {financial_metrics.get('sharpe_ratio', 0.0):.4f}")
        print(f"     最大回撤: {financial_metrics.get('max_drawdown', 0.0):.4f}")
        print(f"     因子换手率: {financial_metrics.get('turnover', 0.0):.4f}")
        print(f"     (原始IC): {financial_metrics.get('rank_ic_mean', 0.0):.4f}")

        print("-" * 25)


if __name__ == "__main__":
    # (用户交互菜单部分保持不变)
    factor_menu = {
        "1": "动量因子", "2": "波动率因子", "3": "情绪/另类因子",
        "4": "价值因子", "5": "质量因子",
        "6": "成长因子", "7": "不指定类型"
    }
    print("请选择您想生成的初始Alpha因子类型:")
    for key, value in factor_menu.items():
        print(f"  {key}: {value}")
    choice = input("请输入选项编号 (默认为7): ")
    selected_type = factor_menu.get(choice, factor_menu["7"])
    print(f"\n好的, 已选择生成: {selected_type}\n")
    run_search(factor_type=selected_type)