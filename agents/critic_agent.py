import json
from typing import Dict, Any
from agents.base_agent import BaseAgent
from utils.data_structures import AlphaFormula


class CriticAgent(BaseAgent):
    """评估一个 alpha 的过拟合风险。"""

    def execute(self, formula: AlphaFormula, history: str) -> Dict[str, Any]:
        """
        执行评估任务。

        Args:
            formula (AlphaFormula): 要评估的 Alpha 公式。
            history (str): 该 Alpha 的优化历史记录。

        Returns:
            一个包含分数和理由的字典。
        """
        formula_str = json.dumps(formula.formula_steps, indent=2)

        formatted_prompt = self.prompt_template.format(
            alpha_formula=formula_str,
            refinement_history=history
        )
        return self._call_llm(formatted_prompt)
