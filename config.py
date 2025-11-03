import os

OPENAI_API_KEY = "sk-fad0tLlqZp0HyxWT6c6cDa8dD9754c71A8329dEa51D1C2Ec"
BASE_URL = "https://openai.wokaai.cn/v1/"
LLM_MODEL = "gpt-4o"

# [修改] 调低了温度，让LLM更“严谨”，减少幻觉
LLM_TEMPERATURE = 0.5

# 是否显示详细的Agent辩论过程
SHOW_DEBATE_LOG = False # 默认关闭，设为 True 可查看辩论详情

# --- MCTS树搜索参数 ---
INITIAL_SEARCH_BUDGET = 10
BUDGET_INCREMENT = 1
MCTS_ITERATIONS = 50
MCTS_EXPLORATION_WEIGHT = 1.414

# --- 评测的五个维度 ---
EFFECTIVENESS_THRESHOLD = 1.8
EVALUATION_DIMENSIONS = [
    "Effectiveness",
    "Stability",
    "Turnover",
    "Diversity",
    "Overfitting Risk"
]
EVAL_TEMP = 1.0
MAX_EVAL_SCORE_PER_DIM = 10.0

AVAILABLE_DATA_FIELDS = [
    "open", "high", "low", "close", "volume", "vwap"
]
AVAILABLE_OPERATORS = [
    "ts_mean", "ts_std", "ts_rank", "ts_corr", "ts_delta",
    "rank", "scale",
    "log", "abs", "sign",
    "add", "subtract", "multiply", "divide",
    # [新增] 确保 backtest.py 中的函数也在这里注册
    "ts_sum", "sma", "stddev", "correlation", "covariance",
    "product", "ts_min", "ts_max", "delay", "ts_argmax",
    "ts_argmin", "decay_linear"
]

# [新增] 定义操作符期望的 *参数* 数量 (伪代码中 param=[...] 列表的长度)
OPERATOR_PARAM_COUNT = {
    # 算术
    "add": 0, "subtract": 0, "multiply": 0, "divide": 0,
    # 数学
    "log": 0, "abs": 0, "sign": 0,
    # 截面
    "rank": 0, "scale": 0,
    # 时序 (需要1个参数，即window或period)
    "ts_mean": 1, "ts_std": 1, "ts_rank": 1, "ts_corr": 1, "ts_delta": 1,
    "ts_sum": 1, "sma": 1, "stddev": 1, "correlation": 1, "covariance": 1,
    "product": 1, "ts_min": 1, "ts_max": 1, "delay": 1,
    "ts_argmax": 1, "ts_argmin": 1, "decay_linear": 1
}

# [新增] 定义操作符期望的 *输入* 数量 (伪代码中 input=[...] 列表的长度)
OPERATOR_INPUT_COUNT = {
    # 算术 (二元)
    "add": 2, "subtract": 2, "multiply": 2, "divide": 2,
    # 时序 (一元)
    "ts_mean": 1, "ts_std": 1, "ts_rank": 1, "ts_delta": 1,
    "ts_sum": 1, "sma": 1, "stddev": 1, "product": 1,
    "ts_min": 1, "ts_max": 1, "delay": 1, "ts_argmax": 1,
    "ts_argmin": 1, "decay_linear": 1,
    # 数学/截面 (一元)
    "log": 1, "abs": 1, "sign": 1, "rank": 1, "scale": 1,
    # 时序 (二元)
    "ts_corr": 2, "correlation": 2, "covariance": 2
}