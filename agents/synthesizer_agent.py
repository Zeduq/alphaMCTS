from typing import Dict, Any, List
from agents.base_agent import BaseAgent
import json
from config import AVAILABLE_DATA_FIELDS, AVAILABLE_OPERATORS


class SynthesizerAgent(BaseAgent):
    """
    一个负责总结辩论结果并生成最终优化后因子画像的Agent。
    """

    def execute(self,
                original_alpha_portrait: Dict[str, Any],
                optimization_target: str,
                factor_type: str,
                fsa_avoid_list: List[str],
                debate_history: str) -> Dict[str, Any]:
        """
        执行辩论总结和画像生成任务。
        """
        try:
            original_portrait_str = json.dumps(original_alpha_portrait, indent=2, ensure_ascii=False)

            formatted_prompt = self.prompt_template.format(
                original_alpha_portrait=original_portrait_str,
                optimization_target=optimization_target,
                factor_type=factor_type,
                fsa_avoid_list=str(fsa_avoid_list),
                debate_history=debate_history,
                available_fields=str(AVAILABLE_DATA_FIELDS),  # <-- 新增
                available_operators=str(AVAILABLE_OPERATORS)  # <-- 新增
            )

            response = self._call_llm(formatted_prompt)

            if isinstance(response,
                          dict) and "name" in response and "description" in response and "pseudo_code" in response and isinstance(
                    response["pseudo_code"], list):
                return response
            else:
                print(f"错误: SynthesizerAgent返回格式无效: {response}")
                return None
        except Exception as e:
            # 增加错误追踪打印，便于调试
            print(f"SynthesizerAgent执行时出错: {e}")
            import traceback
            traceback.print_exc()
            return None