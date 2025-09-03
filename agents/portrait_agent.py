from typing import Dict, Any
from agents.base_agent import BaseAgent
from config import AVAILABLE_DATA_FIELDS, AVAILABLE_OPERATORS

class PortraitAgent(BaseAgent):
    """生成一个概念性的 alpha 画像。"""
    def execute(self) -> Dict[str, Any]:
        """
        执行画像生成任务。

        Returns:
            一个包含 alpha 名称、描述和伪代码的字典。
        """
        formatted_prompt = self.prompt_template.format(
            available_fields=str(AVAILABLE_DATA_FIELDS),
            available_operators=str(AVAILABLE_OPERATORS),
            freq_subtrees="[]"
        )
        return self._call_llm(formatted_prompt)
