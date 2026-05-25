import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class AlphaFormula:
    name: str
    description: str
    formula_steps: List[Dict[str, Any]]
    arguments: List[Dict[str, Any]]

    INVALID_OP_MARKER = "INVALID_OPERATION"
    EMPTY_FORMULA_MARKER = "EMPTY_FORMULA_STEPS"
    NO_FINAL_FORMULA_MARKER = "NO_FINAL_FORMULA"

    def to_expression_string(self) -> str:
        if not self.formula_steps:
            return self.EMPTY_FORMULA_MARKER

        params = self.arguments[0] if self.arguments else {}
        expressions = {}

        for step in self.formula_steps:
            op_name = step.get("name")
            inputs = step.get("input", [])
            op_params = step.get("param", [])
            output_var = step.get("output")

            param_values = [params.get(p, p) for p in op_params]
            input_exprs = [expressions.get(i, i) for i in inputs]

            if op_name in ["add", "subtract", "multiply", "divide"]:
                if len(input_exprs) == 2:
                    op_symbol = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}[op_name]
                    current_expr = f"({input_exprs[0]} {op_symbol} {input_exprs[1]})"
                else:
                    current_expr = self.INVALID_OP_MARKER
            else:
                all_args = input_exprs + param_values
                current_expr = f"{op_name}({', '.join(map(str, all_args))})"

            expressions[output_var] = current_expr

        final_output_var = self.formula_steps[-1].get("output")
        return expressions.get(final_output_var, self.NO_FINAL_FORMULA_MARKER)


@dataclass
class AlphaNode:
    formula: AlphaFormula
    portrait: Dict[str, Any]
    parent: Optional['AlphaNode'] = None
    children: List['AlphaNode'] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    financial_metrics: Dict[str, float] = field(default_factory=dict)
    q_value: float = 0.0
    visits: int = 0
    refinement_summary: str = "Initial root node."

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def calculate_uct(self, exploration_weight: float) -> float:
        if self.visits == 0:
            return float('inf')

        exploitation_term = self.q_value

        if self.parent is None or self.parent.visits == 0:
            return exploitation_term

        exploration_term = exploration_weight * np.sqrt(
            np.log(self.parent.visits) / self.visits
        )

        return exploitation_term + exploration_term

    def __repr__(self):
        name = self.portrait.get('name', 'untitled_alpha')
        return f"AlphaNode(name='{name}', q_value={self.q_value:.2f}, visits={self.visits})"
