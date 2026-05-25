import matplotlib.pyplot as plt
import numpy as np
import re

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_md_table(md_text, section_name):
    pattern = f"\\*\\*{section_name}\\*\\*.*?\\n\\n(.*?)(?=\\n\\n\\*\\*|\\Z)"
    match = re.search(pattern, md_text, re.DOTALL)
    if not match:
        return None
    
    section_content = match.group(1)
    lines = section_content.strip().split('\n')
    data = []
    
    for line in lines:
        if re.match(r'\|\s*\d+\s*\|', line):
            parts = line.split('|')[1:-1]
            if len(parts) >= 5:
                try:
                    row = [int(parts[1].strip()), int(parts[2].strip()), 
                           int(parts[3].strip()), int(parts[4].strip())]
                    data.append(row)
                except:
                    continue
    
    return np.array(data)

# 读取文件
with open('Exp4_3_Rank.md', 'r', encoding='utf-8') as f:
    content = f.read()

# 解析数据
llm_names = ['DeepSeek R1', 'Gemini 3.1 Pro', 'Claude Sonnet 4.6']
all_data = [
    parse_md_table(content, 'DeepSeek R1'),
    parse_md_table(content, 'Gemini 3.1 Pro'),
    parse_md_table(content, 'Claude Sonnet 4.6')
]

# 计算平均值
llm_means = [np.mean(data, axis=0) for data in all_data]

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))
groups = ['Group A\n(本文方法)', 'Group B\n(Alpha101)', 'Group C\n(GP)', 'Group D\n(AlphaAgent)']
x = np.arange(len(groups))
width = 0.25
colors = ['#4472C4', '#ED7D31', '#70AD47']

for i, (llm, color) in enumerate(zip(llm_names, colors)):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, llm_means[i], width, label=llm, color=color, alpha=0.85)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('平均排名（越低表示可解释性越高）', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=11)
ax.set_ylim(0, 4.5)
ax.set_yticks([1, 2, 3, 4])
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=False, edgecolor='gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('interpretability_ranking.png', dpi=300, bbox_inches='tight')
print("图片已保存: interpretability_ranking.png")
