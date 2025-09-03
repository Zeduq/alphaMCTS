# mcts/search.py
import numpy as np
from typing import Optional
from utils.data_structures import AlphaNode, AlphaFormula
from evaluation.evaluator import simulate_evaluation, get_refinement_dimension
from agents.refiner_agent import RefinerAgent
from agents.formula_agent import FormulaAgent
from config import MCTS_EXPLORATION_WEIGHT

refiner_agent = RefinerAgent(prompt_path="prompts/alpha_refinement.txt")
formula_agent = FormulaAgent(prompt_path="prompts/formula_generation.txt")


class MCTS:
    """
    为 Alpha 挖掘实现蒙特卡洛树搜索算法。
    """

    def __init__(self, root: AlphaNode):
        self.root = root

    def _select_child(self, node: AlphaNode) -> Optional[AlphaNode]:
        """
        使用 UCT 公式选择一个子节点。
        """
        if not node.children:
            return None

        uct_scores = []
        for child in node.children:
            if child.visits == 0:
                uct_scores.append(float('inf'))
                continue

            exploitation_term = child.q_value
            exploration_term = MCTS_EXPLORATION_WEIGHT * np.sqrt(
                np.log(node.visits) / child.visits
            )
            uct_scores.append(exploitation_term + exploration_term)

        return node.children[np.argmax(uct_scores)]

    def select(self) -> AlphaNode:
        """
        通过遍历树来选择一个节点进行扩展。
        """
        current_node = self.root
        while not current_node.is_leaf():
            selected_child = self._select_child(current_node)
            if selected_child is None:
                break
            current_node = selected_child
        return current_node

    def expand(self, node_to_expand: AlphaNode) -> Optional[AlphaNode]:
        """
        通过生成一个新的子 Alpha 来扩展一个节点。
        """
        print(f"\n--- 正在扩展节点: {node_to_expand.portrait.get('name', '未命名')} ---")

        refinement_dim = get_refinement_dimension(node_to_expand.scores)
        print(f"优化目标维度: {refinement_dim}")

        suggestion = f"请优化alpha以提升其 '{refinement_dim}' 维度的得分。"
        new_portrait = refiner_agent.execute(
            original_formula=node_to_expand.formula,
            original_portrait=node_to_expand.portrait,
            suggestions=suggestion
        )
        if not new_portrait:
            print("优化智能体未能生成新的画像，跳过本次扩展。")
            return None

        new_formula = formula_agent.execute(alpha_portrait=new_portrait)
        if not new_formula:
            print("公式智能体未能合成公式，跳过本次扩展。")
            return None

        new_node = AlphaNode(
            formula=new_formula,
            portrait=new_portrait,
            parent=node_to_expand,
            refinement_summary=f"针对 {refinement_dim} 进行优化。新思路: {new_portrait.get('description', '无')}"
        )

        new_scores = simulate_evaluation(new_formula, new_node)
        new_node.scores = new_scores
        new_node.q_value = np.mean(list(new_scores.values())) if new_scores else 0.0

        node_to_expand.children.append(new_node)
        print(f"已创建新节点: '{new_node.portrait.get('name', '未命名')}', Q值为: {new_node.q_value:.2f}")
        return new_node

    def backpropagate(self, node: AlphaNode):
        """
        将结果反向传播到根节点。
        """
        current_node = node
        path = []
        while current_node is not None:
            path.append(current_node)
            current_node = current_node.parent

        for node_in_path in path:
            node_in_path.visits += 1
            if node_in_path.children:
                node_in_path.q_value = max([child.q_value for child in node_in_path.children])

            print(
                f"反向传播至: '{node_in_path.portrait.get('name', '未命名')}', 新访问次数: {node_in_path.visits}, 新Q值: {node_in_path.q_value:.2f}")

