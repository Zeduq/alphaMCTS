import os

# GROQ_API_KEY = "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"
OPENAI_API_KEY = "sk-fad0tLlqZp0HyxWT6c6cDa8dD9754c71A8329dEa51D1C2Ec"
# Groq API 的基础 URL
#BASE_URL = "https://api.groq.com/openai/v1"
BASE_URL = "https://openai.wokaai.cn/v1/"

# Groq提供的免费模型
#LLM_MODEL = "llama3-8b-8192"

LLM_MODEL = "gpt-4o"

# --- MCTS树搜索参数 ---
INITIAL_SEARCH_BUDGET = 3
BUDGET_INCREMENT = 1
MCTS_ITERATIONS = 50
MCTS_EXPLORATION_WEIGHT = 1.414

# --- 评测的五个维度 ---
EFFECTIVENESS_THRESHOLD = 7.0
EVALUATION_DIMENSIONS = [
    "Effectiveness",
    "Stability",
    "Turnover",
    "Diversity",
    "Overfitting Risk"
]
EVAL_TEMP = 1.0
MAX_EVAL_SCORE_PER_DIM = 10.0

"""
AVAILABLE_DATA_FIELDS：可用的基础数据字段

open, high, low, close, volume, vwap

AVAILABLE_OPERATORS：可用于构造因子的操作符，包括：

时间序列类：ts_mean, ts_std, ts_rank, ts_corr, ts_delta

排名与缩放：rank, scale

数学变换：log, abs, sign

算术运算：add, subtract, multiply, divide
"""

AVAILABLE_DATA_FIELDS = [
    "open", "high", "low", "close", "volume", "vwap"
]
AVAILABLE_OPERATORS = [
    "ts_mean", "ts_std", "ts_rank", "ts_corr", "ts_delta",
    "rank", "scale",
    "log", "abs", "sign",
    "add", "subtract", "multiply", "divide"
]
