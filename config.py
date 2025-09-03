"""
# config.py


import os

# --- LLM Provider Configuration ---
# 建议使用环境变量来管理API密钥
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
# 为方便测试，暂时保留硬编码，但请注意安全
# 请将 "YOUR_API_KEY_HERE" 替换为您的有效 OpenAI API 密钥
OPENAI_API_KEY = "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"

# 修正: 将模型名称更新为当前广泛可用的 gpt-4o
# 原来的 'gpt-4-turbo' 可能已不被支持或您的账户无权限访问
LLM_MODEL = "gpt-4o"

# --- MCTS Configuration ---
INITIAL_SEARCH_BUDGET = 3
BUDGET_INCREMENT = backtest
MCTS_ITERATIONS = 50
MCTS_EXPLORATION_WEIGHT = backtest.414

# --- Evaluation Configuration ---
EFFECTIVENESS_THRESHOLD = 7.0
EVALUATION_DIMENSIONS = [
    "Effectiveness",
    "Stability",
    "Turnover",
    "Diversity",
    "Overfitting Risk"
]
EVAL_TEMP = backtest.0
MAX_EVAL_SCORE_PER_DIM = 10.0

# --- Alpha Generation Configuration ---
AVAILABLE_DATA_FIELDS = [
    "open", "high", "low", "close", "volume", "vwap"
]
AVAILABLE_OPERATORS = [
    "ts_mean", "ts_std", "ts_rank", "ts_corr", "ts_delta",
    "rank", "scale",
    "log", "abs", "sign",
    "add", "subtract", "multiply", "divide"
]
"""
# config.py

import os

# GROQ_API_KEY = "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"
OPENAI_API_KEY = "sk-fad0tLlqZp0HyxWT6c6cDa8dD9754c71A8329dEa51D1C2Ec"
# Groq API 的基础 URL
#BASE_URL = "https://api.groq.com/openai/v1"
BASE_URL = "https://openai.wokaai.cn/v1/"

# 使用 Groq 提供的免费模型
#LLM_MODEL = "llama3-8b-8192"

LLM_MODEL = "gpt-4o"

# --- MCTS Configuration ---
INITIAL_SEARCH_BUDGET = 3
BUDGET_INCREMENT = 1
MCTS_ITERATIONS = 50
MCTS_EXPLORATION_WEIGHT = 1.414

# --- Evaluation Configuration ---
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

# --- Alpha Generation Configuration ---
AVAILABLE_DATA_FIELDS = [
    "open", "high", "low", "close", "volume", "vwap"
]
AVAILABLE_OPERATORS = [
    "ts_mean", "ts_std", "ts_rank", "ts_corr", "ts_delta",
    "rank", "scale",
    "log", "abs", "sign",
    "add", "subtract", "multiply", "divide"
]
