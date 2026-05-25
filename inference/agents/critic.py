import json
from typing import Dict, Any
from inference.agents.base import BaseAgent
from search.tree import AlphaFormula


class CriticAgent(BaseAgent):
    def execute(self, formula: AlphaFormula, history: str) -> Dict[str, Any]:
        formula_str = json.dumps(formula.formula_steps, indent=2)

        formatted_prompt = self.prompt_template.format(
            alpha_formula=formula_str,
            refinement_history=history
        )
        return self._call_llm(formatted_prompt)
