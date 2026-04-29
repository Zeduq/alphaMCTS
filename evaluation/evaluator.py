"""
Alpha因子评估器模块

注意: 此模块已重构，使用 Evaluator 类封装所有逻辑。
为了向后兼容，保留了原函数接口。
"""

# 导入新的评估器类
from evaluation.evaluator_class import (
    Evaluator,
    EvaluationMetrics,
    get_default_evaluator,
    simulate_evaluation,
    get_refinement_dimension,
)

# 导出公共接口
__all__ = [
    'Evaluator',
    'EvaluationMetrics', 
    'simulate_evaluation',
    'get_refinement_dimension',
    'get_default_evaluator',
]
