import os

# --- 动态配置读取 ---
# 默认为 'gpt-4o'，如果环境变量里有设置（由 run_experiment.py 传入），则使用环境变量
CURRENT_MODEL_TYPE = os.getenv("LLM_MODEL_TYPE", "qwen3-max")
# 默认为 'prompts' (英文)，如果有设置则用设置值 (如 'prompts_cn')
PROMPT_DIR = os.getenv("PROMPT_DIR", "prompts_cn")

print(f"--- [Config] 当前模型模式: {CURRENT_MODEL_TYPE} | 提示词目录: {PROMPT_DIR} ---")

# --- 模型与 API 配置 ---
if CURRENT_MODEL_TYPE == "qwen3-max":
    # [Group C & D 配置]
    # 请在此处填入你的 阿里云 DashScope API Key
    OPENAI_API_KEY = "sk-68e101b794484ae2a1567a066ac8e133"
    # 阿里云兼容 OpenAI 格式的 Base URL
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL = "qwen3-max"  # 或者 "qwen-plus"，根据你购买的模型填写

else:
    # [Group A & B 配置] (默认)
    # 请在此处填入你的 OpenAI / 中转 API Key
    OPENAI_API_KEY = "sk-fad0tLlqZp0HyxWT6c6cDa8dD9754c71A8329dEa51D1C2Ec"
    BASE_URL = "https://openai.wokaai.cn/v1/"
    LLM_MODEL = "gpt-4o"

# --- 其他参数 ---
LLM_TEMPERATURE = 1.0
SHOW_DEBATE_LOG = False  # 建议开启，方便在日志中观察中文/英文辩论的区别

# MCTS 参数
INITIAL_SEARCH_BUDGET = 10
BUDGET_INCREMENT = 1
MCTS_EXPLORATION_WEIGHT = 1.414

# 评测维度
EFFECTIVENESS_THRESHOLD = 2
EVALUATION_DIMENSIONS = [
    "Effectiveness", "Stability", "Turnover", "Diversity", "Overfitting Risk"
]
EVAL_TEMP = 1.0
MAX_EVAL_SCORE_PER_DIM = 10.0

# 字段与算子 (保持不变)
AVAILABLE_DATA_FIELDS = ["open", "high", "low", "close", "volume", "vwap"]
AVAILABLE_OPERATORS = [
    "ts_mean", "ts_std", "ts_rank", "ts_corr", "ts_delta",
    "rank", "scale", "log", "abs", "sign",
    "add", "subtract", "multiply", "divide",
    "ts_sum", "sma", "stddev", "correlation", "covariance",
    "product", "ts_min", "ts_max", "delay", "ts_argmax",
    "ts_argmin", "decay_linear"
]

OPERATOR_PARAM_COUNT = {
    "add": 0, "subtract": 0, "multiply": 0, "divide": 0,
    "log": 0, "abs": 0, "sign": 0, "rank": 0, "scale": 0,
    "ts_mean": 1, "ts_std": 1, "ts_rank": 1, "ts_corr": 1, "ts_delta": 1,
    "ts_sum": 1, "sma": 1, "stddev": 1, "correlation": 1, "covariance": 1,
    "product": 1, "ts_min": 1, "ts_max": 1, "delay": 1,
    "ts_argmax": 1, "ts_argmin": 1, "decay_linear": 1
}

OPERATOR_INPUT_COUNT = {
    "add": 2, "subtract": 2, "multiply": 2, "divide": 2,
    "ts_mean": 1, "ts_std": 1, "ts_rank": 1, "ts_delta": 1,
    "ts_sum": 1, "sma": 1, "stddev": 1, "product": 1,
    "ts_min": 1, "ts_max": 1, "delay": 1, "ts_argmax": 1,
    "ts_argmin": 1, "decay_linear": 1,
    "log": 1, "abs": 1, "sign": 1, "rank": 1, "scale": 1,
    "ts_corr": 2, "correlation": 2, "covariance": 2
}