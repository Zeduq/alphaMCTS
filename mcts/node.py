# mcts_core/node.py

import numpy as np
from typing import Optional, List, Dict, Any


class MCTSNode:
    """
    蒙特卡洛树搜索中的节点类。
    每个节点代表一个特定的Alpha因子及其评估状态。
    """

    def __init__(self,
                 parent: Optional['MCTSNode'],
                 alpha_formula: str,
                 alpha_portrait: Dict[str, Any] = None,
                 scores: Dict[str, float] = None,
                 is_terminal: bool = False):
        """
        初始化一个节点。

        Args:
            parent (MCTSNode | None): 父节点。根节点的父节点为None。
            alpha_formula (str): 该节点代表的Alpha因子公式。
            alpha_portrait (dict, optional): 描述alpha的画像，包含名称、描述等。
            scores (dict, optional): 该Alpha在各个维度的评估分数。
            is_terminal (bool): 标记这是否是一个无法再优化的终结节点。
        """
        self.parent = parent
        self.children: List['MCTSNode'] = []

        self.alpha_formula = alpha_formula
        self.alpha_portrait = alpha_portrait if alpha_portrait else {}

        # MCTS核心统计数据
        self.visit_count = 0
        self.total_score = 0.0  # 根据论文，这里存储子树中的最大分数

        # 评估结果
        self.scores = scores if scores else {}  # e.g., {'effectiveness': 0.7, 'stability': 0.5, ...}

        # 节点的扩展状态
        self.is_fully_expanded = is_terminal
        # 记录已经尝试过的优化动作（维度），避免重复
        self.tried_actions = set()

    @property
    def q_value(self) -> float:
        """
        计算节点的Q值（平均分数）。在我们的框架中，我们更关心最大分数，
        但保留Q值用于标准的UCT计算。
        """
        if self.visit_count == 0:
            return 0
        return self.total_score / self.visit_count

    def uct_score(self, exploration_weight: float = 1.414) -> float:
        """
        计算UCT (Upper Confidence Bound for Trees)分数，用于选择阶段。
        这个分数平衡了利用（exploitation）和探索（exploration）。
        """
        if self.visit_count == 0:
            return float('inf')  # 优先探索未访问过的节点

        # 利用项：节点的历史表现（Q值）
        exploitation_term = self.q_value

        # 探索项：鼓励访问次数较少的节点
        exploration_term = exploration_weight * np.sqrt(
            np.log(self.parent.visit_count) / self.visit_count
        )

        return exploitation_term + exploration_term

    def add_child(self, child_node: 'MCTSNode'):
        """添加一个子节点到当前节点。"""
        self.children.append(child_node)

    def __repr__(self):
        return (f"MCTSNode(formula='{self.alpha_formula}', "
                f"visits={self.visit_count}, score={self.total_score:.4f}, "
                f"children={len(self.children)})")