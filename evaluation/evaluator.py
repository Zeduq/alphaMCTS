# evaluation/evaluator.py
import numpy as np
import pandas as pd
import random
from typing import Dict, Optional, List

# 导入您导师的回测脚本和我们项目的模块
from factor_backtest import FactorBacktest
from utils.data_structures import AlphaFormula, AlphaNode
from agents.critic_agent import CriticAgent
from config import MAX_EVAL_SCORE_PER_DIM, EVAL_TEMP


# ==============================================================================
def get_refinement_dimension(scores: Dict[str, float]) -> str:
    """
    根据分数选择一个维度进行优化。
    """
    refinable_dims = {k: v for k, v in scores.items() if k != "Overfitting Risk"}
    if not refinable_dims:
        return random.choice(list(scores.keys()))
    improvement_scores = np.array([MAX_EVAL_SCORE_PER_DIM - v for v in refinable_dims.values()])
    exp_scores = np.exp(improvement_scores / EVAL_TEMP)
    probabilities = exp_scores / (np.sum(exp_scores) + 1e-9)
    return np.random.choice(list(refinable_dims.keys()), p=probabilities)


# ==============================================================================

critic_agent = CriticAgent(prompt_path="prompts/overfitting_assessment.txt")

try:
    tutor_backtester = FactorBacktest()
    print("✅ 导师的回测引擎初始化成功。")
except Exception as e:
    print(f"CRITICAL ERROR: 初始化 FactorBacktest 失败。请确保 tradelearn 库配置正确。")
    print(f"Error details: {e}")
    raise


class TutorEvaluator:
    """
    一个封装了导师回测代码的评估器。
    """

    def _get_refinement_history(self, node: AlphaNode) -> str:
        history: List[str] = []
        curr = node
        while curr:
            history.append(f"-> {curr.refinement_summary}")
            curr = curr.parent
        return "\n".join(reversed(history))

    def _translate_metrics_to_scores(self, metrics: pd.Series) -> Dict[str, float]:
        scores = {}
        rank_ic = metrics.get('ic_rank_mean', 0.0)
        turnover = metrics.get('turnover_mean', 1.0)
        sharpe = metrics.get('sharpe', 0.0)
        scores["Effectiveness"] = max(0, min(10, abs(rank_ic) * 200))
        scores["Stability"] = max(0, min(10, (sharpe + 2) * 2.5))
        scores["Turnover"] = max(0, min(10, (1 - min(turnover, 1.0)) * 10))
        scores["Diversity"] = round(np.random.uniform(4.0, 10.0), 2)
        return scores

    def evaluate(self, formula: AlphaFormula, node: AlphaNode) -> Optional[Dict[str, float]]:
        expression = formula.to_expression_string()
        print(f"⚙️  正在使用导师的回测引擎评估Alpha: {expression}")

        try:
            factor_name = formula.name
            # === FIX: Corrected the method name here ===
            eval_metrics = tutor_backtester.run_backtest_by_eval(expression, factor_name)

            if eval_metrics is None or eval_metrics.empty:
                print("⚠️ 警告: 回测引擎返回空结果。")
                return None

            scores = self._translate_metrics_to_scores(eval_metrics)
            history_str = self._get_refinement_history(node)
            critic_output = critic_agent.execute(formula=formula, history=history_str)
            if critic_output and 'score' in critic_output:
                scores["Overfitting Risk"] = float(critic_output.get('score', 5.0))
            else:
                scores["Overfitting Risk"] = 5.0

            print(
                f"📊 评估完成. RankIC: {eval_metrics.get('ic_rank_mean', 'N/A'):.4f}, Sharpe: {eval_metrics.get('sharpe', 'N/A'):.4f}")
            return scores

        except Exception as e:
            print(f"❌ 使用导师引擎评估时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None