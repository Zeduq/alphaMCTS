# alphaMCTS/evaluation/evaluator.py

import numpy as np
import pandas as pd
import random
from typing import Dict, List
from utils.data_structures import AlphaFormula, AlphaNode
from agents.critic_agent import CriticAgent
from config import MAX_EVAL_SCORE_PER_DIM, EVAL_TEMP
from factor_backtest import FactorBacktest
from alpha_library.library import AlphaLibrary  # 导入AlphaLibrary

# 在模块级别初始化, 避免每次评估都重新加载数据
critic_agent = CriticAgent(prompt_path="prompts/overfitting_assessment.txt")
backtester = FactorBacktest()


def get_refinement_dimension(scores: Dict[str, float]) -> str:
    """
    根据分数选择一个维度进行优化。
    分数越低的维度被选中的概率越高。
    """
    refinable_dims = {k: v for k, v in scores.items() if k != "Overfitting Risk"}

    if not refinable_dims:
        return random.choice(list(scores.keys()))

    improvement_scores = np.array([MAX_EVAL_SCORE_PER_DIM - v for v in refinable_dims.values()])
    probabilities = np.exp(improvement_scores / EVAL_TEMP) / np.sum(np.exp(improvement_scores / EVAL_TEMP))
    return np.random.choice(list(refinable_dims.keys()), p=probabilities)


def _get_refinement_history(node: AlphaNode) -> str:
    """
    为 Critic Agent 构建节点的优化历史记录字符串。
    """
    history: List[str] = []
    curr = node
    while curr:
        history.append(f"-> {curr.refinement_summary}")
        curr = curr.parent
    return "\n".join(reversed(history))


def calculate_diversity_score(new_factor_values: pd.DataFrame, alpha_repo: AlphaLibrary) -> float:
    """
    计算新因子的多样性得分。
    """
    if not alpha_repo.alphas or new_factor_values is None:
        # 如果库为空或新因子计算失败，则多样性最高
        return MAX_EVAL_SCORE_PER_DIM

    max_corr = 0
    # 遍历库中所有已存在的alpha
    for existing_alpha_data in alpha_repo.alphas:
        formula_obj = existing_alpha_data.get("formula")
        if formula_obj:
            # 计算旧因子的值
            existing_factor_values = backtester.calculate_factor(formula_obj.to_expression_string())
            if existing_factor_values is not None:
                # 计算新旧因子值的截面相关性的绝对值的均值
                corr = new_factor_values.corrwith(existing_factor_values, axis=1).abs().mean()
                if corr > max_corr:
                    max_corr = corr

    # 最大相关性越低，得分越高
    diversity_score = MAX_EVAL_SCORE_PER_DIM * (1 - max_corr)
    return diversity_score


# !! 注意: 我们需要修改 simulate_evaluation 的函数签名，让它可以接收 alpha_repository
def simulate_evaluation(formula: AlphaFormula, node: AlphaNode, alpha_repo: AlphaLibrary) -> Dict[str, float]:
    """
    通过真实回测对一个 alpha 公式的多维度评估。
    """
    scores: Dict[str, float] = {}

    formula_str = formula.to_expression_string()
    # 先计算因子值，因为多样性评分和回测都需要它
    new_factor_values = backtester.calculate_factor(formula_str)
    backtest_results = backtester.run_backtest(formula_str, value_name=formula.name)

    if backtest_results:
        rank_ic = backtest_results.get('rank_ic_mean', 0.0)
        scores["Effectiveness"] = min(MAX_EVAL_SCORE_PER_DIM, abs(rank_ic) * 50)

        icir = backtest_results.get('icir', 0.0)
        scores["Stability"] = min(MAX_EVAL_SCORE_PER_DIM, abs(icir) * 10)

        turnover = backtest_results.get('turnover', 1.0)
        scores["Turnover"] = max(0.0, MAX_EVAL_SCORE_PER_DIM * (1 - turnover))
    else:
        scores["Effectiveness"] = 0.0
        scores["Stability"] = 0.0
        scores["Turnover"] = 0.0

    # --- 新的多样性评分逻辑 ---
    scores["Diversity"] = calculate_diversity_score(new_factor_values, alpha_repo)

    # Overfitting Risk (过拟合风险): 从 Critic Agent 获取
    history_str = _get_refinement_history(node)
    critic_output = critic_agent.execute(formula=formula, history=history_str)

    if critic_output and 'score' in critic_output:
        scores["Overfitting Risk"] = float(critic_output.get('score', 5.0))
        node.refinement_summary += f" | Critic: {critic_output.get('reason', 'N/A')}"
    else:
        scores["Overfitting Risk"] = 5.0

    for k, v in scores.items():
        try:
            scores[k] = round(float(v), 2)
        except (ValueError, TypeError):
            scores[k] = 0.0

    return scores