from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from config import AVAILABLE_DATA_FIELDS, AVAILABLE_OPERATORS

class PortraitAgent(BaseAgent):
    """生成一个概念性的 alpha 画像。"""
    def execute(self, freq_subtrees: List[str], factor_type: str = "综合型") -> Dict[str, Any]:
        """
        执行画像生成任务。

        Args:
            freq_subtrees (List[str]): 一个包含应避免使用的子树表达式的列表。
            factor_type (str): 希望生成的因子类型, 如 "动量因子", "波动率因子" 等。

        Returns:
            一个包含 alpha 名称、描述和伪代码的字典。
        """

        # 定义不同因子类型的指导性Prompt
        guidance_map = {
            "价值因子": "Your design should focus on creating a Value factor. Since we only have price/volume data, try to approximate value concepts, for example by identifying price divergence from a moving average (e.g., vwap).",
            "动量因子": "Your design should focus on creating a Momentum factor, which captures the continuation of price trends.",
            "质量因子": "Your design should focus on creating a Quality factor. Since we lack financial data, approximate 'quality' using price stability or low volatility.",
            "成长因子": "Your design should focus on creating a Growth factor. With only price/volume data, you can interpret rapid price increases combined with volume as a proxy for growth.",
            "波动率因子": "Your design should focus on creating a Volatility factor, which measures the magnitude of price fluctuations.",
            "情绪/另类因子": "Your design should focus on creating a Sentiment or Alternative factor by combining price and volume data in novel ways to represent market activity or crowd behavior.",
            "综合型": "Your design can be of any type. Please create a novel and effective alpha factor based on your expertise."
        }

        factor_type_guidance = guidance_map.get(factor_type, guidance_map["综合型"])
        formatted_prompt = self.prompt_template.format(
            factor_type_guidance=factor_type_guidance,
            available_fields=str(AVAILABLE_DATA_FIELDS),
            available_operators=str(AVAILABLE_OPERATORS),
            # 将列表转换为字符串以填入prompt
            freq_subtrees=str(freq_subtrees)
        )
        return self._call_llm(formatted_prompt)