import numpy as np
import pandas as pd
import random
from typing import Dict, List
from utils.data_structures import AlphaFormula, AlphaNode
from agents.critic_agent import CriticAgent
from config import MAX_EVAL_SCORE_PER_DIM, EVAL_TEMP
from factor_backtest import FactorBacktest
from alpha_library.library import AlphaLibrary

# 在模块级别初始化
critic_agent = CriticAgent(prompt_path="prompts/overfitting_assessment.txt")
backtester = FactorBacktest()


def get_refinement_dimension(scores: Dict[str, float]) -> str:
    """
    根据分数选择一个维度进行优化。
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
    if not alpha_repo.alphas or new_factor_values is None or new_factor_values.empty:
        return MAX_EVAL_SCORE_PER_DIM

    max_corr = 0
    for existing_alpha_data in alpha_repo.alphas:
        formula_obj = existing_alpha_data.get("formula")
        if formula_obj:
            existing_factor_values = backtester.calculate_factor(formula_obj.to_expression_string())
            if existing_factor_values is not None and not existing_factor_values.empty:
                # 对齐索引
                common_index = new_factor_values.index.intersection(existing_factor_values.index)
                if common_index.empty:
                    continue
                new_vals_aligned = new_factor_values.loc[common_index]
                old_vals_aligned = existing_factor_values.loc[common_index]

                # 计算每日横截面相关性的均值
                daily_corr = new_vals_aligned.corrwith(old_vals_aligned, axis=1)
                corr = daily_corr.abs().mean()
                if not np.isnan(corr) and corr > max_corr:
                    max_corr = corr

    diversity_score = MAX_EVAL_SCORE_PER_DIM * (1 - max_corr)
    return diversity_score


def simulate_evaluation(formula: AlphaFormula, node: AlphaNode, alpha_repo: AlphaLibrary) -> Dict[str, float]:
    """
    通过真实回测对一个 alpha 公式的多维度评估。
    """
    scores: Dict[str, float] = {}
    formula_str = formula.to_expression_string()

    # 1. 执行回测，获取所有金融指标
    backtest_results = backtester.run_backtest(formula_str, value_name=formula.name)

    # 2. 将原始金融指标存储到节点上
    if backtest_results:
        node.financial_metrics = backtest_results
    else:
        # 如果回测失败，填充默认失败值
        node.financial_metrics = {
            "rank_ic_mean": 0.0, "rank_ic_std": 0.0, "icir": 0.0, "turnover": 1.0,
            "annualized_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": -1.0
        }

    # 3. 基于金融指标计算5维MCTS分数
    metrics = node.financial_metrics
    rank_ic = metrics.get('rank_ic_mean', 0.0)
    scores["Effectiveness"] = min(MAX_EVAL_SCORE_PER_DIM, abs(rank_ic) * 50)

    icir = metrics.get('icir', 0.0)
    scores["Stability"] = min(MAX_EVAL_SCORE_PER_DIM, abs(icir) * 10)

    turnover = metrics.get('turnover', 1.0)
    scores["Turnover"] = max(0.0, MAX_EVAL_SCORE_PER_DIM * (1 - turnover))

    # 4. 计算多样性 (需要先计算因子值)
    new_factor_values = backtester.calculate_factor(formula_str)  # 重复计算了一次，可优化但目前保持清晰
    scores["Diversity"] = calculate_diversity_score(new_factor_values, alpha_repo)

    # 5. 计算过拟合风险 (保持不变)
    history_str = _get_refinement_history(node)
    critic_output = critic_agent.execute(formula=formula, history=history_str)
    if critic_output and 'score' in critic_output:
        scores["Overfitting Risk"] = float(critic_output.get('score', 5.0))
        node.refinement_summary += f" | Critic: {critic_output.get('reason', 'N/A')}"
    else:
        scores["Overfitting Risk"] = 5.0

    # 6. 格式化分数并返回
    for k, v in scores.items():
        try:
            scores[k] = round(float(v), 2)
        except (ValueError, TypeError):
            scores[k] = 0.0

    return scores