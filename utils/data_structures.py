import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class AlphaFormula:
    """
    代表一个结构化的、机器可读的alpha公式。
    """
    name: str
    description: str
    formula_steps: List[Dict[str, Any]]
    arguments: List[Dict[str, Any]]

    def to_expression_string(self) -> str:
        """
        将结构化的 formula_steps 转换为单行数学表达式字符串。
        """
        if not self.formula_steps:
            return "公式步骤为空"

        # 使用第一组参数进行转换
        params = self.arguments[0] if self.arguments else {}

        # 用于存储每个中间变量的字符串表达式
        expressions = {}

        for step in self.formula_steps:
            op_name = step.get("name")
            inputs = step.get("input", [])
            op_params = step.get("param", [])
            output_var = step.get("output")

            # 从参数字典中获取真实参数值
            param_values = [params.get(p, p) for p in op_params]

            # 获取输入的字符串表达式，如果找不到则直接使用输入名
            input_exprs = [expressions.get(i, i) for i in inputs]

            # 根据操作符类型构建表达式
            current_expr = ""
            if op_name in ["add", "subtract", "multiply", "divide"]:
                # 处理二元运算符
                if len(input_exprs) == 2:
                    op_symbol = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}[op_name]
                    current_expr = f"({input_exprs[0]} {op_symbol} {input_exprs[1]})"
                else:
                    current_expr = "无效的二元运算"
            else:
                # 处理函数式操作符
                all_args = input_exprs + param_values
                current_expr = f"{op_name}({', '.join(map(str, all_args))})"

            expressions[output_var] = current_expr

        # 最后一个步骤的输出就是最终的公式
        final_output_var = self.formula_steps[-1].get("output")
        return expressions.get(final_output_var, "未能生成最终公式")

@dataclass
class AlphaNode:
    """
    代表MCTS搜索树中的一个节点。
    此类已整合 MCTSNode 的功能。
    """
    formula: AlphaFormula
    portrait: Dict[str, Any]
    parent: Optional['AlphaNode'] = None
    children: List['AlphaNode'] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    q_value: float = 0.0
    visits: int = 0
    refinement_summary: str = "Initial root node."

    def is_leaf(self) -> bool:
        """检查节点是否为叶子节点（没有子节点）。"""
        return len(self.children) == 0

    def calculate_uct(self, exploration_weight: float) -> float:
        """
        计算此节点的UCT分数。
        父节点会使用这个分数来决定选择哪个子节点。
        """
        if self.visits == 0:
            return float('inf')  # 优先探索未访问过的节点

        # 利用项：节点的当前Q值
        exploitation_term = self.q_value

        # 探索项：鼓励访问探索次数较少的节点
        if self.parent is None or self.parent.visits == 0:
            # 对于子节点来说不应发生，但作为安全保障
            return exploitation_term

        exploration_term = exploration_weight * np.sqrt(
            np.log(self.parent.visits) / self.visits
        )

        return exploitation_term + exploration_term

    def __repr__(self):
        # 使用 portrait 中的 name 以获得更有意义的表示
        name = self.portrait.get('name', 'untitled_alpha')
        return f"AlphaNode(name='{name}', q_value={self.q_value:.2f}, visits={self.visits})"