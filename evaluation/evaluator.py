# evaluation/evaluator.py
import numpy as np
import random
from typing import Dict, List
from utils.data_structures import AlphaFormula, AlphaNode
from agents.critic_agent import CriticAgent
from config import MAX_EVAL_SCORE_PER_DIM, EVAL_TEMP

# 在模块级别初始化 Agent
critic_agent = CriticAgent(prompt_path="prompts/overfitting_assessment.txt")


def get_refinement_dimension(scores: Dict[str, float]) -> str:
    """
    根据分数选择一个维度进行优化。
    分数越低的维度被选中的概率越高。
    """
    # 从可优化的维度中排除“Overfitting Risk”
    refinable_dims = {k: v for k, v in scores.items() if k != "Overfitting Risk"}

    if not refinable_dims:
        # 如果没有可优化的维度，则随机返回一个（作为备用方案）
        return random.choice(list(scores.keys()))

    # 根据论文公式 P_i(s) = Softmax((e_max * 1_q - E_s) / T)_i
    # 计算每个维度的“改进潜力”分数
    improvement_scores = np.array([MAX_EVAL_SCORE_PER_DIM - v for v in refinable_dims.values()])

    # 使用 softmax 和温度参数计算概率
    probabilities = np.exp(improvement_scores / EVAL_TEMP) / np.sum(np.exp(improvement_scores / EVAL_TEMP))

    # 根据计算出的概率随机选择一个维度
    return np.random.choice(list(refinable_dims.keys()), p=probabilities)


def _get_refinement_history(node: AlphaNode) -> str:
    """
    为 Critic Agent 构建节点的优化历史记录字符串。
    """
    # 修正: 'history =' 是一个语法错误。应初始化为空列表。
    history: List[str] = []
    curr = node
    while curr:
        history.append(f"-> {curr.refinement_summary}")
        curr = curr.parent
    return "\n".join(reversed(history))


def simulate_evaluation(formula: AlphaFormula, node: AlphaNode) -> Dict[str, float]:
    """
    模拟对一个 alpha 公式的多维度评估，这里暂时没有实现。
    """
    scores: Dict[str, float] = {}

    # 模拟量化分数
    scores["Effectiveness"] = round(random.uniform(3.0, 9.5), 2)

    scores["Stability"] = round(random.uniform(2.0, 8.0), 2)

    # 基于回看周期模拟换手率分数
    # 周期越长，换手率越低，分数越高
    try:
        all_params = [p for arg_set in formula.arguments for p in arg_set.values() if isinstance(p, (int, float))]
        avg_period = sum(all_params) / len(all_params) if all_params else 30
    except (TypeError, ZeroDivisionError):
        avg_period = 30  # 默认值
    scores["Turnover"] = round(min(10.0, 10.0 * (60.0 / max(avg_period, 1.0))), 2)

    # 模拟多样性分数 (简化)
    scores["Diversity"] = round(random.uniform(4.0, 10.0), 2)

    # 从 Critic Agent 获取定性分数
    history_str = _get_refinement_history(node)
    critic_output = critic_agent.execute(formula=formula, history=history_str)

    if critic_output and 'score' in critic_output:
        scores["Overfitting Risk"] = float(critic_output.get('score', 5.0))
        # 更新节点的优化摘要，加入来自 Critic 的反馈
        node.refinement_summary += f" | Critic: {critic_output.get('reason', 'N/A')}"
    else:
        scores["Overfitting Risk"] = 5.0  # 失败时的默认分数

    return scores
