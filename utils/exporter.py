import os
import json
import pandas as pd
from datetime import datetime


def export_elite_factors(elite_factors, mode="debate", save_dir="results"):
    """
    将入库的优质因子导出为 JSON 和 CSV (双轨制包含所有指标)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    json_data, csv_data = [], []

    for rank, node in enumerate(elite_factors):
        portrait = node.portrait if hasattr(node, 'portrait') and node.portrait else {}

        # 安全提取公式 (规避自定义对象引发的 JSON 序列化报错)
        formula_out = []
        if hasattr(node, 'formula') and node.formula:
            if hasattr(node.formula, 'formula_steps'):
                formula_out = node.formula.formula_steps
            else:
                formula_out = str(node.formula)

        # 统一使用 financial_metrics 和正确的 scores 大小写
        metrics = getattr(node, 'financial_metrics', {})
        scores = getattr(node, 'scores', {})

        # 1. 构建 JSON 全量字典 (👉 新增了详细的 Financial_Metrics 字典)
        json_data.append({
            "Rank": rank + 1,
            "Node_ID": getattr(node, 'node_id', f"node_{rank}"),
            "Mode": mode,
            "Name": portrait.get("name", "Unknown"),
            "Description": portrait.get("description", "No description"),
            "Formula_Steps": formula_out,
            "Financial_Metrics": {
                "Rank_IC_Mean": round(metrics.get("rank_ic_mean", 0), 4),
                "Normal_IC_Mean": round(metrics.get("ic_mean", 0), 4),  # 👉 同步新增普通IC
                "ICIR": round(metrics.get("icir", 0), 4),
                "Turnover": round(metrics.get("turnover", 0), 4),
                "Annualized_Return": round(metrics.get("annualized_return", 0), 4),
                "Sharpe_Ratio": round(metrics.get("sharpe_ratio", 0), 4),
                "Max_Drawdown": round(metrics.get("max_drawdown", 0), 4)
            },
            "Five_Dimensions_Scores": scores,
            "Q_Value": round(getattr(node, 'q_value', 0.0), 4)
        })

        # 2. 构建 CSV 扁平字典 (👉 确保包含 Normal_IC_Mean)
        csv_data.append({
            "Rank": rank + 1,
            "Factor_ID": getattr(node, 'node_id', f"node_{rank}"),
            "Mode": mode,
            "Name": portrait.get("name", "Unknown"),
            "Rank_IC_Mean": round(metrics.get("rank_ic_mean", 0), 4),
            "Normal_IC_Mean": round(metrics.get("ic_mean", 0), 4),  # 👉 同步新增普通IC
            "ICIR": round(metrics.get("icir", 0), 4),
            "Turnover": round(metrics.get("turnover", 0), 4),
            "Effectiveness": scores.get("Effectiveness", 0),
            "Stability": scores.get("Stability", 0),
            "Turnover_Score": scores.get("Turnover", 0),
            "Diversity": scores.get("Diversity", 0),
            "Overfitting_Risk": scores.get("Overfitting Risk", 0),
            "Q_Value": round(getattr(node, 'q_value', 0.0), 4)
        })

    # 保存 JSON
    json_path = os.path.join(save_dir, f"elite_factors_{mode}_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    # 保存 CSV
    csv_path = os.path.join(save_dir, f"metrics_summary_{mode}_{timestamp}.csv")
    pd.DataFrame(csv_data).to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"\n✅ [{mode.upper()} 模式] 挖掘完毕！共双轨式保存 {len(elite_factors)} 个入库因子。")
    print(f"📄 JSON: {json_path}\n📊 CSV: {csv_path}\n")