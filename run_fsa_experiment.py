import pandas as pd
import numpy as np
import time
from mcts.search import MCTS
from utils.data_structures import AlphaNode, AlphaFormula
from alpha_library.library import AlphaLibrary
from fsa.fsa_miner import mine_frequent_subtrees
from evaluation.evaluator import calculator, simulate_evaluation


def run_ablation_pipeline(use_fsa: bool, iterations: int = 30) -> AlphaLibrary:
    print(f"\n{'=' * 40}")
    print(f"🚀 开始实验组: FSA机制 {'启用' if use_fsa else '禁用'}")
    print(f"{'=' * 40}")

    # 1. 初始化根节点 (使用简单的动量作为起点)
    root_formula = AlphaFormula(
        name="Root_Momentum",
        description="基础动量因子",
        formula_steps=[
            {"name": "delay", "input": ["close"], "param": [10], "output": "delay_close"},
            {"name": "divide", "input": ["close", "delay_close"], "param": [], "output": "ret"},
            {"name": "ts_mean", "input": ["ret"], "param": [5], "output": "factor"}
        ],
        arguments=[{"delay_period": 10, "mean_period": 5}]
    )
    root_portrait = {"name": "Root_Momentum", "factor_type": "动量型", "description": "基础"}
    root_node = AlphaNode(formula=root_formula, portrait=root_portrait)


    alpha_repo = AlphaLibrary()

    print("--- 正在对初始根节点进行回测基线评估 ---")
    root_scores = simulate_evaluation(root_formula, root_node, alpha_repo)
    root_node.scores = root_scores
    root_node.q_value = np.mean(list(root_scores.values())) if root_scores else 0.0
    print(f"根节点初始得分: {root_scores}")

    mcts_tree = MCTS(root=root_node)

    # 2. 开始 MCTS 迭代
    for i in range(iterations):
        print(f"\n--- 迭代 {i + 1}/{iterations} ---")

        # 动态获取规避列表
        freq_subtrees = []
        if use_fsa and len(alpha_repo) > 3:
            # 开启 FSA 且因子库有一定数量时，提取 Top-3 频繁子树
            freq_subtrees = mine_frequent_subtrees(alpha_repo.alphas, top_k=3)

        # 选择与扩展
        node_to_expand = mcts_tree.select()
        new_node = mcts_tree.expand(
            node_to_expand=node_to_expand,
            freq_subtrees=freq_subtrees,
            alpha_repo=alpha_repo,
            ablation_mode="debate"  # 使用完整多智能体模式
        )

        # 反向传播与入库
        if new_node:
            mcts_tree.backpropagate(new_node)
            if new_node.q_value > 0.4:  # 假设 > 0.4 为精英因子
                alpha_repo.add(new_node)

    return alpha_repo


def calculate_library_metrics(alpha_repo: AlphaLibrary):
    if len(alpha_repo) < 2:
        return 0.0, 0.0, 0.0

    # 1. 计算平均 Rank IC
    ics = [a['financial_metrics'].get('rank_ic_mean', 0) for a in alpha_repo.alphas]
    avg_ic = np.mean(ics)

    # 2. 计算横截面平均相关性
    factor_panels = []
    for a in alpha_repo.alphas:
        try:
            factor_df = calculator.calculate_factor(a['formula'].to_expression_string())
            if isinstance(factor_df, pd.Series):
                factor_df = factor_df.unstack()
            factor_panels.append(factor_df)
        except:
            pass

    avg_corr = 0.0
    if len(factor_panels) >= 2:
        corr_matrix = []
        for i in range(len(factor_panels)):
            for j in range(i + 1, len(factor_panels)):
                # 对齐时间轴后计算每日相关性，再求时序均值
                common_idx = factor_panels[i].index.intersection(factor_panels[j].index)
                if not common_idx.empty:
                    daily_corr = factor_panels[i].loc[common_idx].corrwith(factor_panels[j].loc[common_idx], axis=1)
                    corr_matrix.append(daily_corr.abs().mean())
        if corr_matrix:
            avg_corr = np.nanmean(corr_matrix)

    # 3. 提取 Top-1 子树频率
    top1_freq = 0.0
    if len(alpha_repo) > 0:
        from fsa.fsa_miner import _extract_subtrees, Counter
        all_subtrees = []
        for a in alpha_repo.alphas:
            all_subtrees.extend(_extract_subtrees(a['formula']))
        if all_subtrees:
            counts = Counter(all_subtrees)
            top1_freq = counts.most_common(1)[0][1] / len(alpha_repo)

    return avg_ic, avg_corr, top1_freq


if __name__ == "__main__":
    # 运行对照组 (无 FSA)
    repo_baseline = run_ablation_pipeline(use_fsa=False, iterations=10)
    ic_base, corr_base, freq_base = calculate_library_metrics(repo_baseline)

    # 运行实验组 (有 FSA)
    repo_fsa = run_ablation_pipeline(use_fsa=True, iterations=10)
    ic_fsa, corr_fsa, freq_fsa = calculate_library_metrics(repo_fsa)

    # 输出结果报告
    print("\n" + "=" * 50)
    print("🎯 FSA 消融实验结果报告")
    print("=" * 50)
    print(f"{'指标':<25} | {'w/o FSA (对照组)':<15} | {'w/ FSA (实验组)':<15}")
    print("-" * 55)
    print(f"{'精英因子产出数量':<21} | {len(repo_baseline):<15} | {len(repo_fsa):<15}")
    print(f"{'Top-1 AST 结构复用率':<18} | {freq_base * 100:.2f}%{'':<9} | {freq_fsa * 100:.2f}%")
    print(f"{'库内平均横截面相关性 (ρ)':<17} | {corr_base:.4f}{'':<9} | {corr_fsa:.4f}")
    print(f"{'精英库平均 Rank IC':<19} | {ic_base:.4f}{'':<9} | {ic_fsa:.4f}")
    print("=" * 50)