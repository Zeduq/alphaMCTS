import json
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from utils.data_structures import AlphaFormula
from config import AVAILABLE_DATA_FIELDS, AVAILABLE_OPERATORS

class RefinerAgent(BaseAgent):
    """根据指导性建议优化一个现有的 alpha。"""
    def execute(self, original_formula: AlphaFormula, original_portrait: Dict[str, Any], suggestions: str, freq_subtrees: List[str]) -> Dict[str, Any]:
        """
        执行优化任务。

        Args:
            original_formula (AlphaFormula): 原始的 Alpha 公式对象。
            original_portrait (Dict[str, Any]): 原始的 Alpha 画像。
            suggestions (str): 用于指导优化的建议。
            freq_subtrees (List[str]): 一个包含应避免使用的子树表达式的列表。

        Returns:
            一个新的 alpha 画像字典。
        """
        origin_alpha_str = json.dumps(original_portrait, indent=2)

        formatted_prompt = self.prompt_template.format(
            available_fields=str(AVAILABLE_DATA_FIELDS),
            available_operators=str(AVAILABLE_OPERATORS),
            freq_subtrees=str(freq_subtrees),
            origin_alpha_formula=origin_alpha_str,
            refinement_suggestions=suggestions
        )
        return self._call_llm(formatted_prompt)