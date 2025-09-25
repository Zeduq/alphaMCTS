from collections import Counter
from typing import List, Dict, Any
from utils.data_structures import AlphaFormula


def _extract_subtrees(formula: AlphaFormula) -> List[str]:
    """
    从单个AlphaFormula中提取简化的子树表达式。

    根据论文，我们关注操作符和输入的结构，忽略具体参数。
    例如: ts_mean(close, 20) -> "ts_mean(close)"
    """
    subtrees = set()
    # 存储每个变量是由哪个表达式生成的
    expressions = {}

    for step in formula.formula_steps:
        op_name = step.get("name")
        inputs = step.get("input", [])
        output_var = step.get("output")

        # 将输入变量替换为它们的来源表达式（如果可用），否则直接使用输入名
        input_exprs = [expressions.get(i, i) for i in inputs]

        # 简化表达式，只包含操作符和输入
        # 例如: "add(close, open)", "ts_rank(vwap)"
        current_expr = f"{op_name}({', '.join(sorted(input_exprs))})"

        expressions[output_var] = current_expr
        subtrees.add(current_expr)

    return list(subtrees)


def mine_frequent_subtrees(alpha_library: List[Dict[str, Any]], top_k: int = 3) -> List[str]:
    """
    从Alpha库中挖掘最常见的top_k个子树。

    Args:
        alpha_library (List[Dict[str, Any]]): Alpha仓库中的alpha数据列表。
        top_k (int): 返回最常见的子树数量。

    Returns:
        一个包含top_k个最常见子树表达式字符串的列表。
    """
    if not alpha_library:
        return []

    all_subtrees = []
    for alpha_data in alpha_library:
        formula_obj = alpha_data.get("formula")
        if formula_obj and isinstance(formula_obj, AlphaFormula):
            all_subtrees.extend(_extract_subtrees(formula_obj))

    if not all_subtrees:
        return []

    # 统计每个子树出现的频率
    subtree_counts = Counter(all_subtrees)

    # 找到最常见的 top_k 个子树
    most_common = subtree_counts.most_common(top_k)

    # 返回子树的字符串列表
    frequent_subtrees = [subtree for subtree, count in most_common]

    print(f"--- FSA Miner: 发现最常见的子树: {frequent_subtrees} ---")
    return frequent_subtrees