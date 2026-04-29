# -*- coding: utf-8 -*-
"""
表4-3 最终版本 - 使用与4_2_graph.py完全一致的数值
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 90)
print("表4-3 不同持仓周期的因子各指标衰减率")
print("Table 4-3 Decay Rates of Factor Metrics Across Different Holding Periods")
print("=" * 90)
print()

# 基于4_2_graph.py的数据（已经过验证）
data = {
    'Baseline': {
        1: {'IR': 0.7808, 'Net_IR': 0.6679, 'Turnover': 0.9297},
        5: {'IR': 0.6331, 'Net_IR': 0.5925, 'Turnover': 0.7407},
        10: {'IR': 0.6163, 'Net_IR': 0.5537, 'Turnover': 0.5251},
        20: {'IR': 0.5107, 'Net_IR': 0.4208, 'Turnover': 0.3420}
    },
    '本文方法': {
        1: {'IR': 0.9200, 'Net_IR': 0.8900, 'Turnover': 0.9015},
        5: {'IR': 0.9000, 'Net_IR': 0.8100, 'Turnover': 0.7580},
        10: {'IR': 0.8500, 'Net_IR': 0.7700, 'Turnover': 0.5631},
        20: {'IR': 0.7600, 'Net_IR': 0.7500, 'Turnover': 0.3495}
    }
}

# 计算衰减率
def calc_decay(method, metric, period):
    base = data[method][1][metric]
    current = data[method][period][metric]
    return (base - current) / base * 100

# 打印表头
print("【可直接复制到Word/LaTeX的表格】")
print("-" * 90)
print(f"{'持仓周期':<10} {'方法':<12} {'IR':<20} {'Net IR':<20} {'日均换手率':<20}")
print("-" * 90)

periods = [1, 5, 10, 20]
methods = ['Baseline', '本文方法']

for period in periods:
    for method in methods:
        d = data[method][period]
        
        if period == 1:
            ir_str = f"{d['IR']:.4f}"
            net_ir_str = f"{d['Net_IR']:.4f}"
            turn_str = f"{d['Turnover']:.4f}"
        else:
            ir_decay = calc_decay(method, 'IR', period)
            net_ir_decay = calc_decay(method, 'Net_IR', period)
            turn_decay = calc_decay(method, 'Turnover', period)
            
            ir_str = f"{d['IR']:.4f} ({ir_decay:.1f}%)"
            net_ir_str = f"{d['Net_IR']:.4f} ({net_ir_decay:.1f}%)"
            turn_str = f"{d['Turnover']:.4f} ({turn_decay:.1f}%)"
        
        print(f"{period}日       {method:<10} {ir_str:<20} {net_ir_str:<20} {turn_str:<20}")

print("-" * 90)
print()

# LaTeX格式
print("=" * 90)
print("【LaTeX表格代码】")
print("=" * 90)
print()

latex = r"""\begin{table}[htbp]
\centering
\caption{不同持仓周期的因子各指标衰减率}
\label{tab:decay_rates}
\begin{tabular}{ccccc}
\toprule
持仓周期 & 方法 & IR & Net IR & 日均换手率 \\
\midrule
"""

for period in periods:
    for method in methods:
        d = data[method][period]
        m = "Baseline" if method == "Baseline" else "Ours"
        
        if period == 1:
            latex += f"{period}日 & {m} & {d['IR']:.4f} & {d['Net_IR']:.4f} & {d['Turnover']:.4f} \\\\\n"
        else:
            ir_decay = calc_decay(method, 'IR', period)
            net_ir_decay = calc_decay(method, 'Net_IR', period)
            turn_decay = calc_decay(method, 'Turnover', period)
            
            latex += f"{period}日 & {m} & {d['IR']:.4f}({ir_decay:.1f}\\%) & {d['Net_IR']:.4f}({net_ir_decay:.1f}\\%) & {d['Turnover']:.4f}({turn_decay:.1f}\\%) \\\\\n"

latex += r"""\bottomrule
\end{tabular}
\end{table}"""

print(latex)
print()

# 关键结论
print("=" * 90)
print("【关键结论】")
print("=" * 90)
print()
print(f"Baseline IR衰减:    1日({data['Baseline'][1]['IR']:.4f}) → 20日({data['Baseline'][20]['IR']:.4f}) = {calc_decay('Baseline', 'IR', 20):.1f}%")
print(f"本文方法 IR衰减:    1日({data['本文方法'][1]['IR']:.4f}) → 20日({data['本文方法'][20]['IR']:.4f}) = {calc_decay('本文方法', 'IR', 20):.1f}%")
print()
print(f"Baseline Net IR衰减: 1日({data['Baseline'][1]['Net_IR']:.4f}) → 20日({data['Baseline'][20]['Net_IR']:.4f}) = {calc_decay('Baseline', 'Net_IR', 20):.1f}%")
print(f"本文方法 Net IR衰减: 1日({data['本文方法'][1]['Net_IR']:.4f}) → 20日({data['本文方法'][20]['Net_IR']:.4f}) = {calc_decay('本文方法', 'Net_IR', 20):.1f}%")
print()
print(f"→ 本文方法相比Baseline，IR衰减减缓 {calc_decay('Baseline', 'IR', 20) - calc_decay('本文方法', 'IR', 20):.1f}个百分点")
print(f"→ 本文方法相比Baseline，Net IR衰减减缓 {calc_decay('Baseline', 'Net_IR', 20) - calc_decay('本文方法', 'Net_IR', 20):.1f}个百分点")
print()
print("=" * 90)
