import json
from typing import Dict, Any, Optional
from inference.agents.base import BaseAgent
from search.tree import AlphaFormula
from config import AVAILABLE_DATA_FIELDS, AVAILABLE_OPERATORS


class FormulaAgent(BaseAgent):
    def execute(self, alpha_portrait: Dict[str, Any]) -> Optional[AlphaFormula]:
        formatted_prompt = self.prompt_template.format(
            available_fields=str(AVAILABLE_DATA_FIELDS),
            available_operators=str(AVAILABLE_OPERATORS),
            alpha_portrait_prompt=json.dumps(alpha_portrait, indent=2),
            window_range="(5, 60)"
        )

        structured_formula_json = self._call_llm(formatted_prompt)

        if structured_formula_json and 'formula' in structured_formula_json and 'arguments' in structured_formula_json:
            return AlphaFormula(
                name=alpha_portrait.get('name', 'untitled_alpha'),
                description=alpha_portrait.get('description', 'No description.'),
                formula_steps=structured_formula_json['formula'],
                arguments=structured_formula_json['arguments']
            )

        print("Formula Agent failed: LLM回复格式非法或存在参数缺失")
        return None
