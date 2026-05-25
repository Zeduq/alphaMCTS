from collections import Counter
from typing import List, Dict, Any
from search.tree import AlphaFormula


def _extract_subtrees(formula: AlphaFormula) -> List[str]:
    subtrees = set()
    expressions = {}

    for step in formula.formula_steps:
        op_name = step.get("name")
        inputs = step.get("input", [])
        output_var = step.get("output")

        input_exprs = [expressions.get(i, i) for i in inputs]
        current_expr = f"{op_name}({', '.join(sorted(input_exprs))})"

        expressions[output_var] = current_expr
        subtrees.add(current_expr)

    return list(subtrees)


def mine_frequent_subtrees(alpha_library: List[Dict[str, Any]], top_k: int = 3) -> List[str]:
    if not alpha_library:
        return []

    all_subtrees = []
    for alpha_data in alpha_library:
        formula_obj = alpha_data.get("formula")
        if formula_obj and isinstance(formula_obj, AlphaFormula):
            all_subtrees.extend(_extract_subtrees(formula_obj))

    if not all_subtrees:
        return []

    subtree_counts = Counter(all_subtrees)
    most_common = subtree_counts.most_common(top_k)
    frequent_subtrees = [subtree for subtree, count in most_common]

    print(f"--- FSA Miner: 发现最常见的子树: {frequent_subtrees} ---")
    return frequent_subtrees
