import json
import numpy as np
import traceback
from mcts.search import MCTS
from utils.data_structures import AlphaNode, AlphaFormula
from agents.portrait_agent import PortraitAgent
from agents.formula_agent import FormulaAgent
from evaluation.evaluator import simulate_evaluation
from config import INITIAL_SEARCH_BUDGET, BUDGET_INCREMENT, EFFECTIVENESS_THRESHOLD
from alpha_library.library import AlphaLibrary
from fsa.fsa_miner import mine_frequent_subtrees


def initialize_root_node() -> AlphaNode:
    """
    使用由Agent生成的第一个Alpha来初始化MCTS树的根节点。
    """
    print("--- 正在初始化根节点 ---")
    portrait_agent = PortraitAgent(prompt_path="prompts/portrait_generation.txt")
    formula_agent = FormulaAgent(prompt_path="prompts/formula_generation.txt")

    # 初始生成时，没有需要规避的子树
    root_portrait = portrait_agent.execute(freq_subtrees=[])
    if not root_portrait:
        raise Exception("生成初始Alpha画像失败。")

    root_formula = formula_agent.execute(alpha_portrait=root_portrait)
    if not root_formula:
        raise Exception("合成初始Alpha公式失败。")

    root_node = AlphaNode(formula=root_formula, portrait=root_portrait)

    # 评估根节点时，传入一个空的AlphaLibrary实例，因为此时库里还没有任何因子
    root_scores = simulate_evaluation(root_formula, root_node, AlphaLibrary())
    root_node.scores = root_scores
    root_node.q_value = np.mean(list(root_scores.values())) if root_scores else 0

    # 详细的打印输出 (同步您的改动)
    print("--- 根节点详细信息 ---")
    print(f"根节点 '{root_node.portrait.get('name', '未命名')}' 已创建, Q值为: {root_node.q_value:.2f}")
    # 使用 to_expression_string() 获取可读的公式
    formula_str = root_node.formula.to_expression_string()
    print(f"  因子公式: {formula_str}")
    scores_str = json.dumps(root_node.scores, indent=4)
    print(f"  五维得分:\n{scores_str}")
    print("--------------------")

    return root_node


def run_search():
    """
    运行MCTS搜索过程的主函数。
    """
    try:
        root_node = initialize_root_node()
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

    if root_node.scores.get("Effectiveness", 0) >= EFFECTIVENESS_THRESHOLD:
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

            if new_node.q_value > max_score_overall:
                max_score_overall = new_node.q_value
                search_budget += BUDGET_INCREMENT
                print(f"*** 发现新的最佳Alpha, Q值: {max_score_overall:.2f}。搜索预算增加至: {search_budget} ***")

            if new_node.scores.get("Effectiveness", 0) >= EFFECTIVENESS_THRESHOLD:
                effective_alpha_repository.add(new_node)

        i += 1

    print("\n\n--- MCTS 搜索完成 ---")
    print(f"仓库中有效Alpha总数: {len(effective_alpha_repository)}")

    best_alphas = effective_alpha_repository.get_best_alphas(n=10)
    print(f"\n--- 仓库中排名前 {len(best_alphas)} 的Alpha ---")
    for idx, alpha_data in enumerate(best_alphas):
        portrait = alpha_data.get('portrait', {})
        formula_obj = alpha_data.get('formula')

        scores = alpha_data.get('scores', {})
        scores_str = json.dumps(scores, indent=4)

        print(f"{idx + 1}. 名称: {portrait.get('name', '未命名Alpha')}")
        print(f"   Q值: {alpha_data.get('q_value', 0):.4f}")
        print(f"   描述: {portrait.get('description', '无描述')}")

        if formula_obj and isinstance(formula_obj, AlphaFormula):
            expression_str = formula_obj.to_expression_string()
            print(f"   公式: {expression_str}")
        else:
            # Fallback for older data structure if needed, though unlikely with current code
            formula_steps_count = len(formula_obj.formula_steps) if formula_obj and hasattr(formula_obj,
                                                                                            'formula_steps') else 0
            print(f"   公式步骤: {formula_steps_count} 步")

        # 打印格式化后的五维得分
        print(f"   五维得分:\n{scores_str}")
        print("-" * 25)


if __name__ == "__main__":
    run_search()