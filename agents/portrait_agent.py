from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from config import AVAILABLE_DATA_FIELDS, AVAILABLE_OPERATORS

class PortraitAgent(BaseAgent):
    """生成一个概念性的 alpha 画像。"""
    def execute(self, freq_subtrees: List[str]) -> Dict[str, Any]:
        """
        执行画像生成任务。

        Args:
            freq_subtrees (List[str]): 一个包含应避免使用的子树表达式的列表。

        Returns:
            一个包含 alpha 名称、描述和伪代码的字典。
        """
        formatted_prompt = self.prompt_template.format(
            available_fields=str(AVAILABLE_DATA_FIELDS),
            available_operators=str(AVAILABLE_OPERATORS),
            # 将列表转换为字符串以填入prompt
            freq_subtrees=str(freq_subtrees)
        )
        return self._call_llm(formatted_prompt)