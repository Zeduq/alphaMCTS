from typing import List, Dict, Any
from search.tree import AlphaNode


class AlphaLibrary:
    def __init__(self):
        self.alphas: List[Dict[str, Any]] = []

    def add(self, node: AlphaNode):
        alpha_data = {
            "formula": node.formula,
            "portrait": node.portrait,
            "q_value": node.q_value,
            "scores": node.scores,
            "financial_metrics": node.financial_metrics,
            "visit_count": node.visits
        }
        alpha_name = alpha_data["portrait"].get('name', '未命名')
        self.alphas.append(alpha_data)
        print(f"Alpha '{alpha_name}' 已添加至仓库。当前总数: {len(self.alphas)}")

    def get_best_alphas(self, n: int = 5) -> List[Dict[str, Any]]:
        sorted_alphas = sorted(self.alphas, key=lambda x: x.get('q_value', 0), reverse=True)
        return sorted_alphas[:n]

    def __len__(self):
        return len(self.alphas)
