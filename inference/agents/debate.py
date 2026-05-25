from typing import Dict, Any, List
from inference.agents.base import BaseAgent
import json

from config import AVAILABLE_DATA_FIELDS, AVAILABLE_OPERATORS


class DebateAgent(BaseAgent):
    def execute(self,
                persona: str,
                current_alpha_portrait: Dict[str, Any],
                optimization_target: str,
                factor_type: str,
                fsa_avoid_list: List[str],
                debate_history: str) -> Dict[str, Any]:
        try:
            portrait_str = json.dumps(current_alpha_portrait, indent=2, ensure_ascii=False)

            formatted_prompt = self.prompt_template.format(
                persona=persona,
                current_alpha_portrait=portrait_str,
                optimization_target=optimization_target,
                factor_type=factor_type,
                fsa_avoid_list=str(fsa_avoid_list),
                debate_history=debate_history if debate_history else "No discussion yet. You speak first.",
                available_fields=str(AVAILABLE_DATA_FIELDS),
                available_operators=str(AVAILABLE_OPERATORS)
            )

            response = self._call_llm(formatted_prompt)

            if isinstance(response, dict) and "analysis" in response and "contribution" in response:
                return response
            else:
                print(f"错误: DebateAgent返回格式无效: {response}")
                return None
        except Exception as e:
            print(f"DebateAgent执行时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
