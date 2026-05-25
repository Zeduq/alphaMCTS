import json
from typing import Dict, Any, List
from inference.agents.base import BaseAgent
from search.tree import AlphaFormula
from config import AVAILABLE_DATA_FIELDS, AVAILABLE_OPERATORS


class RefinerAgent(BaseAgent):
    def execute(self, original_formula: AlphaFormula, original_portrait: Dict[str, Any], suggestions: str, freq_subtrees: List[str]) -> Dict[str, Any]:
        origin_alpha_str = json.dumps(original_portrait, indent=2)

        formatted_prompt = self.prompt_template.format(
            available_fields=str(AVAILABLE_DATA_FIELDS),
            available_operators=str(AVAILABLE_OPERATORS),
            freq_subtrees=str(freq_subtrees),
            origin_alpha_formula=origin_alpha_str,
            refinement_suggestions=suggestions
        )
        return self._call_llm(formatted_prompt)
