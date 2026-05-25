import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 定义文件路径
RESULTS_LOG_DIR = "results/logs"
OUTPUT_DIR = "output/figures"

file_paths = {
    "Group A\n(GPT-4o/En)": os.path.join(RESULTS_LOG_DIR, "result_Group_A_GPT4o_English.txt"),
    "Group B\n(GPT-4o/Zh)": os.path.join(RESULTS_LOG_DIR, "result_Group_B_GPT4o_Chinese.txt"),
    "Group C\n(Qwen/En)": os.path.join(RESULTS_LOG_DIR, "result_Group_C_QwenMax_English.txt"),
    "Group D\n(Qwen/Zh)": os.path.join(RESULTS_LOG_DIR, "result_Group_D_QwenMax_Chinese.txt")
}


def parse_result_file(filepath, group_name):
    alphas = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"警告: 找不到文件 {filepath}")
        return []

    sections = content.split('--- Alpha')

    for section in sections[1:]:
        alpha_data = {'Group': group_name}

        # 提取基础信息
        name_match = re.search(r'Name: (.*)', section)
        if name_match: alpha_data['Name'] = name_match.group(1).strip()

        q_match = re.search(r'Q-Value: ([\d\.]+)', section)
        if q_match: alpha_data['Q-Value'] = float(q_match.group(1))

        # 提取 Scores
        scores_match = re.search(r'Scores: ({.*?})', section)
        if scores_match:
            try:
                scores = json.loads(scores_match.group(1))
                for k, v in scores.items():
                    alpha_data[f'Score_{k}'] = v
            except:
                pass

        # 提取 Metrics
        metrics_match = re.search(r'Financial Metrics: ({.*?})', section)
        if metrics_match:
            try:
                metrics = json.loads(metrics_match.group(1))
                for k, v in metrics.items():
                    alpha_data[f'Metric_{k}'] = v
            except:
                pass

        alphas.append(alpha_data)
    return alphas


# 2. 数据加载
print("正在读取数据...")
all_alphas = []
for group_name, path in file_paths.items():
    all_alphas.extend(parse_result_file(path, group_name))

df = pd.DataFrame(all_alphas)

# 3. 设置绘图风格 (美化)
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
# 字体设置：优先尝试微软雅黑/黑体，否则用默认
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 6))
# 统计每个组的数量
count_df = df['Group'].value_counts().reset_index()
count_df.columns = ['Group', 'Count']
# 保持自定义的顺序
order = list(file_paths.keys())

# 使用 pointplot 绘制折线趋势 (更适合展示离散类别的变化趋势)
sns.pointplot(x='Group', y='Count', data=count_df, order=order,
              color='#e74c3c', markers='o', linestyles='-', scale=1.2)

# 添加数字标签
for i, row in count_df.iterrows():
    # 由于数据是乱序的，需要找到对应的x轴位置
    x_idx = order.index(row['Group'])
    plt.text(x_idx, row['Count'] + 0.5, f"{row['Count']}",
             ha='center', va='bottom', fontsize=12, fontweight='bold', color='#e74c3c')

plt.title('各实验组生成有效因子数量对比', fontsize=15, pad=20)
plt.ylabel('生成数量 (个)')
plt.xlabel('')
plt.ylim(0, max(count_df['Count']) + 3)
plt.tight_layout()
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(os.path.join(OUTPUT_DIR, 'factor_counts.png'), dpi=300)
print("已生成: output/figures/factor_counts.png (因子数量折线图)")

plt.figure(figsize=(12, 6))
sns.boxplot(x='Group', y='Q-Value', data=df, hue='Group', palette="Set3", legend=False)
plt.title('各实验组因子综合评分 (Q-Value) 分布', fontsize=15)
plt.ylabel('Q-Value')
plt.xlabel('')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'q_value_distribution.png'), dpi=300)
print("已生成: output/figures/q_value_distribution.png")

plt.figure(figsize=(12, 6))
sns.barplot(x='Group', y='Metric_rank_ic_mean', data=df, hue='Group',
            estimator='mean', errorbar='sd', palette="viridis", legend=False)
plt.title('各实验组平均 Rank IC (预测能力) 及波动率', fontsize=15)
plt.ylabel('Rank IC Mean')
plt.xlabel('')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rank_ic_comparison.png'), dpi=300)
print("已生成: output/figures/rank_ic_comparison.png")

plt.figure(figsize=(12, 7))
sns.scatterplot(x='Metric_turnover', y='Metric_icir', hue='Group', style='Group',
                s=150, data=df, alpha=0.85, palette="deep")

# 添加辅助线和区域标注
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5)  # 假设0.5是换手率中位数
plt.text(0.1, df['Metric_icir'].max(), '理想区域\n(高收益/低换手)', color='green', ha='left', va='top')

plt.title('风险调整收益 (ICIR) vs 换手率 (Turnover)', fontsize=15)
plt.xlabel('Turnover (换手率) -> 越低越好')
plt.ylabel('ICIR (信息比率) -> 越高越好')
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'icir_vs_turnover.png'), dpi=300)
print("已生成: output/figures/icir_vs_turnover.png")

plt.figure(figsize=(12, 6))
sns.stripplot(x='Group', y='Score_Overfitting Risk', data=df, hue='Group',
              size=8, palette="magma", jitter=True, alpha=0.8, legend=False)
plt.title('过拟合风险评分分布 (分数越高 = 风险越低)', fontsize=15)
plt.ylabel('Risk Score (0-10)')
plt.xlabel('')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'overfitting_risk.png'), dpi=300)
print("已生成: output/figures/overfitting_risk.png")

print("\n所有图表分析已完成！")
