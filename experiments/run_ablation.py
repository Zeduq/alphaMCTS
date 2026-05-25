import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import numpy as np
from config import PROMPT_DIR
from search.lifecycle import MCTS
from search.tree import AlphaNode, AlphaFormula
from utils.exporter import export_elite_factors
from constraint.library import AlphaLibrary
from inference.agents.portrait import PortraitAgent
from inference.agents.formula import FormulaAgent
from evaluation.evaluator import simulate_evaluation


def initialize_real_root_node(factor_type: str) -> AlphaNode:
    print(f"\n--- [初始化] 正在生成真实的初始根节点 (领域: {factor_type}) ---")
    portrait_agent = PortraitAgent(prompt_path=os.path.join(PROMPT_DIR, "portrait_generation.txt"))
    formula_agent = FormulaAgent(prompt_path=os.path.join(PROMPT_DIR, "formula_generation.txt"))

    # 1. 生成初始画像
    root_portrait = portrait_agent.execute(freq_subtrees=[], factor_type=factor_type)
    if not root_portrait:
        raise Exception("生成初始Alpha画像失败。")

    # 2. 生成初始公式
    root_formula = formula_agent.execute(alpha_portrait=root_portrait)
    if not root_formula:
        raise Exception("合成初始Alpha公式失败。")

    # 确保公式是 AlphaFormula 对象
    if isinstance(root_formula, dict):
        root_formula = AlphaFormula(**root_formula)

    root_node = AlphaNode(formula=root_formula, portrait=root_portrait)

    # 3. 初始节点评估
    print("--- [初始化] 正在回测评估初始根节点 ---")
    root_scores = simulate_evaluation(root_node.formula, root_node, AlphaLibrary())
    root_node.scores = root_scores
    root_node.q_value = np.mean(list(root_scores.values())) if root_scores else 0.0

    print(f"✅ 真实根节点 '{root_node.portrait.get('name', '未命名')}' 初始化完成! 初始Q值: {root_node.q_value:.2f}")
    return root_node


def run_experiment(mode="refiner", target_count=10, factor_type="动量因子"):
    print(f"============== 开始 {mode.upper()} 消融实验 ==============")
    print(f"============== 指定因子类型: {factor_type} ==============")

    alpha_repo = AlphaLibrary()
    freq_subtrees = []

    try:
        root_node = initialize_real_root_node(factor_type)
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    # 将真实节点作为 MCTS 的根
    mcts = MCTS(root=root_node)
    elite_pool = []
    budget = 100  # 你可以根据需要调大 budget

    # 如果运气极好，初始节点直接就满足入库要求，先把它收了
    if root_node.q_value > 6.0:
        elite_pool.append(root_node)
        print(f"🎉 初始节点极为优秀，直接入库！当前进度: {len(elite_pool)}/{target_count}")

    while len(elite_pool) < target_count and budget > 0:
        print(f"\n[Budget 剩余: {budget} | 已入库: {len(elite_pool)}/{target_count}]")

        # 1. 选择
        node_to_expand = mcts.select()

        # 2. 拓展 (传入消融开关 mode)
        new_node = mcts.expand(node_to_expand, freq_subtrees, alpha_repo, ablation_mode=mode)

        if new_node:
            # 3. 反向传播
            mcts.backpropagate(new_node)

            # 判断是否满足入库标准
            if new_node.q_value > 4.5:
                elite_pool.append(new_node)
                metrics = getattr(new_node, 'financial_metrics', {})
                rank_ic = metrics.get('rank_ic_mean', 0.0)
                norm_ic = metrics.get('ic_mean', 0.0)

                print(f"🎉 成功捕获高分因子！目前入库数量：{len(elite_pool)}/{target_count}")
                print(f"   -> [指标] Rank IC: {rank_ic:.4f} | 普通 IC: {norm_ic:.4f} | Q值: {new_node.q_value:.2f}")

        budget -= 1

    # 执行双轨制导出
    export_elite_factors(elite_pool, mode=mode, save_dir="results")
    print(f"============== {mode.upper()} 实验结束 ==============")


if __name__ == "__main__":
    # 你可以在这里指定想要测试的因子类型和模式
    # 建议两组对照实验（refiner 和 debate）使用相同的 factor_type
    experiment_domain = "动量因子"

    run_experiment(mode="refiner", target_count=10, factor_type=experiment_domain)
    #run_experiment(mode="debate", target_count=10, factor_type=experiment_domain)

