import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表风格
sns.set_style("whitegrid")
sns.set_palette("Set2")

def parse_result_file(file_path):
    """解析结果文件，提取因子信息"""
    factors = []
    current_factor = {}
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 开始解析新因子
        if line.startswith('--- Alpha'):
            if current_factor:
                factors.append(current_factor)
            current_factor = {}
        
        # 解析名称
        elif line.startswith('Name:'):
            current_factor['name'] = line.split(':', 1)[1].strip()
        
        # 解析Q值
        elif line.startswith('Q-Value:'):
            current_factor['q_value'] = float(line.split(':', 1)[1].strip())
        
        # 解析评分
        elif line.startswith('Scores:'):
            scores_str = line.split(':', 1)[1].strip()
            try:
                scores = json.loads(scores_str)
                current_factor.update(scores)
            except json.JSONDecodeError as e:
                print(f"解析评分失败: {e}")
        
        # 解析财务指标
        elif line.startswith('Financial Metrics:'):
            # 财务指标可能跨多行，需要读取完整
            metrics_str = line.split(':', 1)[1].strip()
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('--- Alpha'):
                metrics_str += lines[i].strip()
                i += 1
            i -= 1  # 回退一行，以便下一次循环正确处理
            
            try:
                metrics = json.loads(metrics_str)
                current_factor.update(metrics)
            except json.JSONDecodeError as e:
                print(f"解析财务指标失败: {e}")
        
        i += 1
    
    # 添加最后一个因子
    if current_factor:
        factors.append(current_factor)
    
    return factors

def create_comparison_charts(new_factors, old_factors):
    """创建对比图表"""
    # 创建DataFrame
    new_df = pd.DataFrame(new_factors)
    old_df = pd.DataFrame(old_factors)
    
    # 确保两个DataFrame有相同的因子数量
    min_len = min(len(new_df), len(old_df))
    new_df = new_df.head(min_len)
    old_df = old_df.head(min_len)
    
    # 创建因子索引
    new_df['factor_num'] = range(1, len(new_df) + 1)
    old_df['factor_num'] = range(1, len(old_df) + 1)
    
    # 合并数据
    new_df['version'] = 'New'
    old_df['version'] = 'Old'
    combined_df = pd.concat([new_df, old_df], ignore_index=True)
    
    # 定义要对比的指标
    score_metrics = ['Effectiveness', 'Stability', 'Turnover', 'Diversity', 'Overfitting Risk']
    financial_metrics = ['rank_ic_mean', 'icir', 'excess_return', 'information_ratio', 'sharpe_ratio']
    
    # 创建子图
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    # 绘制评分指标对比
    for i, metric in enumerate(score_metrics[:3]):
        ax = axes[i]
        sns.boxplot(x='version', y=metric, data=combined_df, ax=ax)
        sns.swarmplot(x='version', y=metric, data=combined_df, color='black', size=3, ax=ax)
        ax.set_title(f'{metric} 对比')
        ax.set_ylabel(metric)
        ax.set_xlabel('版本')
    
    # 绘制多样性和过拟合风险
    for i, metric in enumerate(score_metrics[3:5]):
        ax = axes[i+3]
        sns.boxplot(x='version', y=metric, data=combined_df, ax=ax)
        sns.swarmplot(x='version', y=metric, data=combined_df, color='black', size=3, ax=ax)
        ax.set_title(f'{metric} 对比')
        ax.set_ylabel(metric)
        ax.set_xlabel('版本')
    
    # 绘制ICIR和信息比率
    for i, metric in enumerate(['icir', 'information_ratio']):
        ax = axes[i+5]
        sns.boxplot(x='version', y=metric, data=combined_df, ax=ax)
        sns.swarmplot(x='version', y=metric, data=combined_df, color='black', size=3, ax=ax)
        ax.set_title(f'{metric} 对比')
        ax.set_ylabel(metric)
        ax.set_xlabel('版本')
    
    # 绘制IC均值
    ax = axes[7]
    sns.boxplot(x='version', y='rank_ic_mean', data=combined_df, ax=ax)
    sns.swarmplot(x='version', y='rank_ic_mean', data=combined_df, color='black', size=3, ax=ax)
    ax.set_title('Rank IC Mean 对比')
    ax.set_ylabel('Rank IC Mean')
    ax.set_xlabel('版本')
    
    # 绘制多样性与ICIR的散点图
    ax = axes[8]
    sns.scatterplot(x='Diversity', y='icir', hue='version', data=combined_df, ax=ax)
    ax.set_title('多样性与ICIR关系对比')
    ax.set_xlabel('多样性')
    ax.set_ylabel('ICIR')
    
    plt.tight_layout()
    plt.savefig('factor_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建因子排名对比图
    plt.figure(figsize=(12, 8))
    
    # Q值排名对比
    plt.subplot(2, 2, 1)
    sns.lineplot(x='factor_num', y='q_value', hue='version', data=combined_df)
    plt.title('Q值排名对比')
    plt.xlabel('因子排名')
    plt.ylabel('Q值')
    
    # 多样性排名对比
    plt.subplot(2, 2, 2)
    sns.lineplot(x='factor_num', y='Diversity', hue='version', data=combined_df)
    plt.title('多样性排名对比')
    plt.xlabel('因子排名')
    plt.ylabel('多样性分数')
    
    # ICIR排名对比
    plt.subplot(2, 2, 3)
    sns.lineplot(x='factor_num', y='icir', hue='version', data=combined_df)
    plt.title('ICIR排名对比')
    plt.xlabel('因子排名')
    plt.ylabel('ICIR')
    
    # 信息比率排名对比
    plt.subplot(2, 2, 4)
    sns.lineplot(x='factor_num', y='information_ratio', hue='version', data=combined_df)
    plt.title('信息比率排名对比')
    plt.xlabel('因子排名')
    plt.ylabel('信息比率')
    
    plt.tight_layout()
    plt.savefig('factor_ranking_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建箱线图综合对比
    plt.figure(figsize=(12, 10))
    
    # 评分指标箱线图
    plt.subplot(2, 1, 1)
    melted_scores = pd.melt(combined_df, id_vars=['version'], value_vars=score_metrics, var_name='指标', value_name='值')
    sns.boxplot(x='指标', y='值', hue='version', data=melted_scores)
    plt.title('评分指标综合对比')
    plt.xticks(rotation=45)
    plt.legend(title='版本')
    
    # 财务指标箱线图
    plt.subplot(2, 1, 2)
    melted_financial = pd.melt(combined_df, id_vars=['version'], value_vars=financial_metrics, var_name='指标', value_name='值')
    sns.boxplot(x='指标', y='值', hue='version', data=melted_financial)
    plt.title('财务指标综合对比')
    plt.xticks(rotation=45)
    plt.legend(title='版本')
    
    plt.tight_layout()
    plt.savefig('factor_metrics_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return combined_df

def analyze_differences(combined_df):
    """分析差异"""
    new_df = combined_df[combined_df['version'] == 'New']
    old_df = combined_df[combined_df['version'] == 'Old']
    
    print("=== 因子挖掘结果对比分析 ===")
    print("\n1. 基本统计信息:")
    print(f"新版本因子数量: {len(new_df)}")
    print(f"旧版本因子数量: {len(old_df)}")
    
    print("\n2. 关键指标对比 (平均值):")
    key_metrics = ['Diversity', 'icir', 'rank_ic_mean', 'information_ratio', 'sharpe_ratio', 'excess_return']
    for metric in key_metrics:
        new_mean = new_df[metric].mean()
        old_mean = old_df[metric].mean()
        diff = new_mean - old_mean
        pct_change = (diff / old_mean * 100) if old_mean != 0 else 0
        print(f"{metric}: 新版本={new_mean:.4f}, 旧版本={old_mean:.4f}, 差异={diff:.4f} ({pct_change:.2f}%)")
    
    print("\n3. 指标分布对比:")
    for metric in key_metrics:
        new_std = new_df[metric].std()
        old_std = old_df[metric].std()
        print(f"{metric}标准差: 新版本={new_std:.4f}, 旧版本={old_std:.4f}")
    
    print("\n4. 排名前5的因子对比:")
    print("\n新版本前5因子:")
    print(new_df[['name', 'q_value', 'Diversity', 'icir']].head(5))
    print("\n旧版本前5因子:")
    print(old_df[['name', 'q_value', 'Diversity', 'icir']].head(5))

if __name__ == "__main__":
    # 解析结果文件
    new_factors = parse_result_file('result.txt')
    old_factors = parse_result_file('result_old.txt')
    
    print(f"新版本因子数量: {len(new_factors)}")
    print(f"旧版本因子数量: {len(old_factors)}")
    
    # 创建对比图表
    combined_df = create_comparison_charts(new_factors, old_factors)
    
    # 分析差异
    analyze_differences(combined_df)
    
    print("\n对比图表已生成:")
    print("- factor_comparison.png: 各指标散点和箱线图对比")
    print("- factor_ranking_comparison.png: 因子排名对比图")
    print("- factor_metrics_boxplot.png: 指标箱线图综合对比")
    print("\n分析完成!")