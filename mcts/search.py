import numpy as np
from typing import Optional, List
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

        # 使用节点的 calculate_uct 方法作为 key，直接找到分数最高的子节点
        return max(node.children, key=lambda child: child.calculate_uct(MCTS_EXPLORATION_WEIGHT))

    def select(self) -> AlphaNode:
        """
        通过遍历树来选择一个节点进行扩展。
        该实现允许选择内部节点进行扩展，而不仅仅是叶子节点。
        """
        current_node = self.root

        while current_node.children:  # 只要还有子节点，就继续决策
            best_child = self._select_child(current_node)
            best_child_score = best_child.calculate_uct(MCTS_EXPLORATION_WEIGHT)

            # 根据论文4.1节，我们计算一个“虚拟”的自我扩展分数
            # Q值使用当前节点的Q值，访问次数使用子节点数量（论文中的一个简化）
            # 我们增加一个小的epsilon避免子节点数量为0时除零
            virtual_self_visits = max(1, len(current_node.children))
            exploration_term_self = MCTS_EXPLORATION_WEIGHT * np.sqrt(
                np.log(current_node.visits) / virtual_self_visits
            )
            # 自我扩展的分数 = 当前节点的Q值 + 探索项
            self_expansion_score = current_node.q_value + exploration_term_self

            # 如果自我扩展的分数更高，则停止并选择当前节点进行扩展
            if self_expansion_score > best_child_score:
                print(
                    f"决策: 扩展当前节点 '{current_node.portrait.get('name', '未命名')}' (自身分数 {self_expansion_score:.2f} > 最佳子节点分数 {best_child_score:.2f})")
                return current_node

            # 否则，继续向下遍历
            print(
                f"决策: 向下遍历至 '{best_child.portrait.get('name', '未命名')}' (子节点分数 {best_child_score:.2f} 更高)")
            current_node = best_child

        # 如果循环结束（到达叶子节点），则直接选择该叶子节点
        print(f"决策: 到达叶子节点，选择 '{current_node.portrait.get('name', '未命名')}' 进行扩展")
        return current_node

    def expand(self, node_to_expand: AlphaNode, freq_subtrees: List[str]) -> Optional[AlphaNode]:
        """
        通过生成一个新的子 Alpha 来扩展一个节点。
        """
        print(f"\n--- 正在扩展节点: {node_to_expand.portrait.get('name', '未命名')} ---")
        print(f"--- FSA: 当前规避列表: {freq_subtrees} ---")

        refinement_dim = get_refinement_dimension(node_to_expand.scores)
        print(f"优化目标维度: {refinement_dim}")

        suggestion = f"请优化alpha以提升其 '{refinement_dim}' 维度的得分。"
        new_portrait = refiner_agent.execute(
            original_formula=node_to_expand.formula,
            original_portrait=node_to_expand.portrait,
            suggestions=suggestion,
            # 将规避列表传递给 RefinerAgent
            freq_subtrees=freq_subtrees
        )
        if not new_portrait:
            print("优化智能体未能生成新的画像，跳过本次扩展。")
            return None

        # 注意：FormulaAgent目前不需要规避列表，因为它只是翻译画像
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
        根据论文，Q值应为子树中的最大分数。
        """
        current_node = node
        # 新节点自身的分数
        new_score = node.q_value

        while current_node is not None:
            # 增加访问次数
            current_node.visits += 1

            # 更新Q值：如果新分数高于当前节点的Q值，则更新
            if new_score > current_node.q_value:
                current_node.q_value = new_score

            print(
                f"反向传播至: '{current_node.portrait.get('name', '未命名')}', "
                f"新访问次数: {current_node.visits}, "
                f"更新后Q值: {current_node.q_value:.2f}"
            )

            current_node = current_node.parent

