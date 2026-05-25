
import os
from pathlib import Path
from typing import List


def load_dotenv_builtin(env_path: Path):
    if not env_path.exists():
        return
    
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过空行和注释
                if not line or line.startswith('#'):
                    continue
                # 解析 KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # 移除可能的引号
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    # 设置环境变量
                    if key:
                        os.environ[key] = value
    except Exception as e:
        print(f"[Config] 警告: 读取 .env 文件失败: {e}")


# 加载 .env 文件
env_path = Path(__file__).parent / '.env'
load_dotenv_builtin(env_path)

# 尝试使用 python-dotenv（如果已安装）
try:
    from dotenv import load_dotenv
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # 使用内置解析器即可



# 默认为 'qwen3-max'，如果环境变量里有设置，则使用环境变量
CURRENT_MODEL_TYPE = os.getenv("LLM_MODEL_TYPE", "qwen3-max")

# 默认为 'prompts_cn'，如果有设置则用设置值
PROMPT_DIR = os.getenv("PROMPT_DIR", "inference/prompts/cn")

# 数据目录
DATA_DIR = os.getenv("DATA_DIR", "D:/AAProject/Data")


def print_config():
    print(f"--- [Config] 当前模型模式: {CURRENT_MODEL_TYPE} | 提示词目录: {PROMPT_DIR} ---")
    print(f"--- [Config] 搜索预算: {INITIAL_SEARCH_BUDGET} | 准入阈值: {EFFECTIVENESS_THRESHOLD} ---")



def get_api_config():
    if CURRENT_MODEL_TYPE in ("qwen3-max", "qwen-turbo"):
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "未设置 DASHSCOPE_API_KEY 环境变量。\n"
                "请在 .env 文件中设置: DASHSCOPE_API_KEY=your-api-key"
            )
        return {
            "api_key": api_key,
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": CURRENT_MODEL_TYPE
        }
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "未设置 OPENAI_API_KEY 环境变量。\n"
                "请在 .env 文件中设置: OPENAI_API_KEY=your-api-key"
            )
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return {
            "api_key": api_key,
            "base_url": base_url,
            "model": CURRENT_MODEL_TYPE
        }


try:
    api_config = get_api_config()
    OPENAI_API_KEY = api_config["api_key"]
    BASE_URL = api_config["base_url"]
    LLM_MODEL = api_config["model"]
except ValueError as e:
    # 未设置 API Key，使用占位值
    OPENAI_API_KEY = ""
    BASE_URL = ""
    LLM_MODEL = CURRENT_MODEL_TYPE
    print(f"[Config] 警告: {e}")



LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "1.0"))
SHOW_DEBATE_LOG = os.getenv("SHOW_DEBATE_LOG", "False").lower() == "true"



INITIAL_SEARCH_BUDGET = int(os.getenv("INITIAL_SEARCH_BUDGET", "10"))
BUDGET_INCREMENT = int(os.getenv("BUDGET_INCREMENT", "1"))
MCTS_EXPLORATION_WEIGHT = float(os.getenv("MCTS_EXPLORATION_WEIGHT", "1.414"))

# 辩论轮数
DEBATE_ROUNDS = int(os.getenv("DEBATE_ROUNDS", "1"))



EFFECTIVENESS_THRESHOLD = float(os.getenv("EFFECTIVENESS_THRESHOLD", "2.0"))

EVALUATION_DIMENSIONS: List[str] = [
    "Effectiveness", "Stability", "Turnover", "Diversity", "Overfitting Risk"
]
EVAL_TEMP = float(os.getenv("EVAL_TEMP", "1.0"))
MAX_EVAL_SCORE_PER_DIM = 10.0

# 入库Q值阈值
ELITE_Q_THRESHOLD = 5.0



AVAILABLE_DATA_FIELDS: List[str] = ["open", "high", "low", "close", "volume", "vwap"]

AVAILABLE_OPERATORS: List[str] = [
    "ts_mean", "ts_std", "ts_rank", "ts_corr", "ts_delta",
    "rank", "scale", "log", "abs", "sign",
    "add", "subtract", "multiply", "divide",
    "ts_sum", "sma", "stddev", "correlation", "covariance",
    "product", "ts_min", "ts_max", "delay", "ts_argmax",
    "ts_argmin", "decay_linear"
]

# 算子参数数量映射
OPERATOR_PARAM_COUNT = {
    "add": 0, "subtract": 0, "multiply": 0, "divide": 0,
    "log": 0, "abs": 0, "sign": 0, "rank": 0, "scale": 0,
    "ts_mean": 1, "ts_std": 1, "ts_rank": 1, "ts_corr": 1, "ts_delta": 1,
    "ts_sum": 1, "sma": 1, "stddev": 1, "correlation": 1, "covariance": 1,
    "product": 1, "ts_min": 1, "ts_max": 1, "delay": 1,
    "ts_argmax": 1, "ts_argmin": 1, "decay_linear": 1
}

# 算子输入数量映射
OPERATOR_INPUT_COUNT = {
    "add": 2, "subtract": 2, "multiply": 2, "divide": 2,
    "ts_mean": 1, "ts_std": 1, "ts_rank": 1, "ts_delta": 1,
    "ts_sum": 1, "sma": 1, "stddev": 1, "product": 1,
    "ts_min": 1, "ts_max": 1, "delay": 1, "ts_argmax": 1,
    "ts_argmin": 1, "decay_linear": 1,
    "log": 1, "abs": 1, "sign": 1, "rank": 1, "scale": 1,
    "ts_corr": 2, "correlation": 2, "covariance": 2
}



# 训练期（用于因子挖掘）
TRAIN_BEGIN = "2017-01-03"
TRAIN_END = "2022-06-26"

# 测试期（仅用于回测验证）
TEST_BEGIN = "2022-06-27"
TEST_END = "2023-06-26"


if __name__ == "__main__":
    print_config()
    print(f"\nAPI配置: model={LLM_MODEL}, base_url={BASE_URL[:30]}...")

