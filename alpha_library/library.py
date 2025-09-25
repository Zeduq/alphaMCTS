from typing import List, Dict, Any
from utils.data_structures import AlphaNode

class AlphaLibrary:
    """
    用于存储和管理在搜索过程中发现的有效的Alpha因子。
    """
    def __init__(self):
        self.alphas: List[Dict[str, Any]] = []

    def add(self, node: AlphaNode):
        """
        将一个MCTS节点代表的Alpha添加到仓库中。
        """
        alpha_data = {
            "formula": node.formula,
            "portrait": node.portrait,
            "q_value": node.q_value,
            "scores": node.scores,
            "visit_count": node.visits
        }
        alpha_name = alpha_data["portrait"].get('name', '未命名')
        self.alphas.append(alpha_data)
        print(f"✅ Alpha '{alpha_name}' 已添加至仓库。当前总数: {len(self.alphas)}")

    def get_best_alphas(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        根据Q值返回表现最好的n个Alpha。
        """
        sorted_alphas = sorted(self.alphas, key=lambda x: x.get('q_value', 0), reverse=True)
        return sorted_alphas[:n]

    def __len__(self):
        return len(self.alphas)
