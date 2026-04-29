import re
import json
import os


def convert_txt_to_json(txt_path="debate_result.txt", output_json_path="debate_result_converted.json"):
    if not os.path.exists(txt_path):
        print(f"错误：未找到文件 {txt_path}")
        return

    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按 "--- Alpha X ---" 切分文本块
    blocks = re.split(r'--- Alpha \d+ ---', content)

    factors = []
    for block in blocks:
        if not block.strip() or 'Name:' not in block:
            continue

        # 使用正则表达式提取各个字段
        name_match = re.search(r'Name:\s*(.*)', block)
        q_val_match = re.search(r'Q-Value:\s*([\d.]+)', block)
        formula_match = re.search(r'Formula:\s*(.*)', block)
        scores_match = re.search(r'Scores:\s*(\{.*\})', block)
        metrics_match = re.search(r'Financial Metrics:\s*(\{.*\})', block)

        if not (name_match and q_val_match and formula_match and scores_match and metrics_match):
            continue

        name = name_match.group(1).strip()
        q_value = float(q_val_match.group(1).strip())
        formula_str = formula_match.group(1).strip()
        scores = json.loads(scores_match.group(1).strip())
        metrics = json.loads(metrics_match.group(1).strip())

        factors.append({
            "Name": name,
            "Q_Value": q_value,
            "Formula_Steps": formula_str,  # 旧版txt只有公式字符串，直接存入
            "Scores": scores,
            "Metrics": metrics
        })

    # 🌟 核心逻辑：根据 Q_Value 从高到低排序，以便重新计算 Rank
    factors.sort(key=lambda x: x["Q_Value"], reverse=True)

    json_data = []
    for rank, factor in enumerate(factors):
        metrics = factor["Metrics"]
        scores = factor["Scores"]

        # 按照约定好的 JSON 结构组织数据
        json_item = {
            "Rank": rank + 1,  # 自己计算出的排名
            "Node_ID": f"node_placeholder_{rank}",  # 占位符
            "Mode": "debate",  # 占位符
            "Name": factor["Name"],
            "Description": "No description available (Parsed)",  # 占位符
            "Formula_Steps": factor["Formula_Steps"],
            "Financial_Metrics": {
                "Rank_IC_Mean": round(metrics.get("rank_ic_mean", 0), 4),
                # 注意：这里按要求没有放入 Normal_IC_Mean
                "ICIR": round(metrics.get("icir", 0), 4),
                "Turnover": round(metrics.get("turnover", 0), 4),
                "Annualized_Return": round(metrics.get("annualized_return", 0), 4),
                "Sharpe_Ratio": round(metrics.get("sharpe_ratio", 0), 4),
                "Max_Drawdown": round(metrics.get("max_drawdown", 0), 4)
            },
            "Five_Dimensions_Scores": scores,
            "Visits": 1,  # 占位符
            "Q_Value": factor["Q_Value"]
        }
        json_data.append(json_item)

    # 写入 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    print(f"✅ 转换成功！共提取并重新排序了 {len(json_data)} 个因子。")
    print(f"📄 结构化 JSON 已保存至: {output_json_path}")


if __name__ == "__main__":
    # 执行转换
    convert_txt_to_json(txt_path="debate_result.txt", output_json_path="debate_result_converted.json")