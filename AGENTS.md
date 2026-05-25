# AlphaMCTS 项目文档

## 项目概述

AlphaMCTS 是一个基于蒙特卡洛树搜索（MCTS）和多智能体辩论机制的量化投资 Alpha 因子自动挖掘系统。该系统通过 LLM（大语言模型）驱动的智能体协作，自动发现、优化和评估金融市场中的有效 Alpha 因子。

### 核心特性

- **MCTS 搜索框架**：使用蒙特卡洛树搜索探索 Alpha 因子空间
- **多智能体辩论机制**：通过 Agent_A（积极探索）、Agent_B（谨慎保守）、Critic（批判评审）三方辩论优化因子
- **FSA（频繁子树规避）**：自动识别并规避过度使用的因子结构，保证多样性
- **双轨评估体系**：五维评分（有效性、稳定性、换手率、多样性、过拟合风险）+ 金融指标（IC/IR、年化收益、夏普比率等）
- **支持多模型对比**：可同时测试 GPT-4o 和 Qwen-Max 等不同 LLM 的效果

## 技术栈

- **编程语言**：Python 3.x
- **核心依赖**：
  - `openai`：LLM API 调用（支持 OpenAI 格式，包括阿里云 DashScope）
  - `pandas`, `numpy`：数据处理与数值计算
  - `alphalens`：因子分析（可选，降级方案为手动计算）
  - `empyrical`：金融指标计算
  - `scikit-learn`：机器学习（策略回测）
  - `gplearn`：遗传规划基线对比
  - `matplotlib`, `seaborn`：可视化分析
- **回测框架**：trade-learn（项目内嵌）

## 项目结构

```
alphaMCTS/
├── search/                 # 搜索模块：树的建模、生命周期管理
│   ├── tree.py             # AlphaFormula, AlphaNode 数据结构
│   └── lifecycle.py        # MCTS 实现（选择、扩展、反向传播）
├── constraint/             # 约束模块：FSA 规避、因子库维护
│   ├── fsa.py              # 频繁子树挖掘
│   └── library.py          # AlphaLibrary 有效因子存储与管理
├── inference/              # 推理模块：智能体协作规范与实现
│   ├── agents/             # LLM 智能体
│   │   ├── base.py         # 智能体抽象基类
│   │   ├── portrait.py     # Alpha 画像生成智能体
│   │   ├── formula.py      # 公式翻译智能体
│   │   ├── debate.py       # 辩论智能体
│   │   ├── synthesizer.py  # 辩论结果合成智能体
│   │   ├── critic.py       # 过拟合评估智能体
│   │   └── refiner.py      # 因子优化智能体（消融实验用）
│   └── prompts/            # 提示词模板
│       ├── en/             # 英文提示词
│       └── cn/             # 中文提示词
├── evaluation/             # 评估模块
│   ├── evaluator.py        # 兼容层
│   └── evaluator_class.py  # 因子回测与五维评分核心实现
├── utils/                  # 工具模块
│   ├── data_structures.py  # 兼容层（重新导出 search.tree）
│   └── exporter.py         # 结果导出（JSON/CSV 双轨制）
├── tools/                  # 数据预处理工具（部分实现）
│   ├── select.py, scale.py, neutralize.py, label.py
│   ├── encode.py, balance.py, outlier.py, miss.py
├── experiments/            # 实验脚本
│   ├── run_experiment.py   # 2x2 对照实验运行器
│   ├── run_ablation.py     # 消融实验（refiner vs debate）
│   ├── run_fsa_experiment.py # FSA 机制消融实验
│   ├── run_gp_baseline.py  # 遗传规划基线对比
│   ├── analyze_results.py  # 实验结果可视化分析
│   ├── compare_results.py  # 新旧版本对比分析
│   ├── recalculate_metrics.py # 指标重新计算工具
│   └── convert_txt_to_json.py # 结果格式转换工具
├── trade-learn-master/     # 内嵌回测框架
├── config.py               # 全局配置（模型、API、参数）
├── main.py                 # 主程序入口
├── factor_calculator.py    # 因子计算公式解析与计算
├── strategy.py             # 回测策略定义
└── mcts_gui.py             # 可视化 GUI 系统
```

## 配置说明

### 配置文件：`config.py`

关键配置项：

```python
# 模型配置（通过环境变量动态切换）
CURRENT_MODEL_TYPE = os.getenv("LLM_MODEL_TYPE", "qwen3-max")  # 或 "gpt-4o"
PROMPT_DIR = os.getenv("PROMPT_DIR", "inference/prompts/cn")  # 或 "inference/prompts/en"

# API 配置（需填入有效 API Key）
OPENAI_API_KEY = "your-api-key"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 阿里云
# 或 BASE_URL = "https://openai.wokaai.cn/v1/"  # OpenAI 中转

# MCTS 参数
INITIAL_SEARCH_BUDGET = 3       # 初始搜索预算
BUDGET_INCREMENT = 1            # 发现优质因子后的预算增量
MCTS_EXPLORATION_WEIGHT = 1.414 # UCT 探索权重

# 评测维度
EFFECTIVENESS_THRESHOLD = 1     # 有效性入库阈值
EVALUATION_DIMENSIONS = ["Effectiveness", "Stability", "Turnover", "Diversity", "Overfitting Risk"]

# 可用数据字段与算子
AVAILABLE_DATA_FIELDS = ["open", "high", "low", "close", "volume", "vwap"]
AVAILABLE_OPERATORS = ["ts_mean", "ts_std", "ts_rank", "ts_corr", "ts_delta", ...]
```

### 数据路径配置

在 `evaluation/evaluator.py` 和 `factor_calculator.py` 中配置：

```python
DATA_DIR = 'D:/AAProject/Data'  # 数据根目录
# 需要的数据文件：
# - 000300SH.csv    # 沪深300成分股行情数据
# - hs300_index.csv # 沪深300指数（基准）
# - alpha101.csv    # Alpha101 因子库（多样性对比用）
```

## 运行方式

### 1. 主程序运行

```bash
python main.py
```

交互式选择因子类型（1-7）：
1. 动量因子
2. 波动率因子
3. 情绪/另类因子
4. 价值因子
5. 质量因子
6. 成长因子
7. 不指定类型

### 2. 2x2 对照实验

```bash
python experiments/run_experiment.py
```

自动运行 4 组实验：
- Group A: GPT-4o + 英文提示词
- Group B: GPT-4o + 中文提示词
- Group C: Qwen-Max + 英文提示词
- Group D: Qwen-Max + 中文提示词（重点关注）

### 3. 消融实验

```bash
# Refiner 模式（单智能体优化）
python experiments/run_ablation.py

# FSA 机制对比
python experiments/run_fsa_experiment.py
```

### 4. 结果分析

```bash
# 生成对比图表
python experiments/analyze_results.py

# 新旧版本对比
python experiments/compare_results.py

# 重新计算指标
python experiments/recalculate_metrics.py
```

## 代码规范

### 命名规范

- **类名**：PascalCase（如 `AlphaFormula`, `PortraitAgent`）
- **函数/方法名**：snake_case（如 `calculate_factor`, `execute`）
- **变量名**：snake_case，私有变量前缀下划线
- **常量名**：全大写（如 `INITIAL_SEARCH_BUDGET`, `AVAILABLE_OPERATORS`）

### 类型注解

鼓励使用类型注解：

```python
from typing import Dict, Any, List, Optional

def execute(self, alpha_portrait: Dict[str, Any]) -> Optional[AlphaFormula]:
    ...
```

### 文档字符串

类和方法应包含中文文档字符串：

```python
class AlphaFormula:
    """
    代表一个结构化的、机器可读的alpha公式。
    """
    
    def to_expression_string(self) -> str:
        """
        将结构化的 formula_steps 转换为单行数学表达式字符串。
        """
```

### 错误处理

使用 try-except 捕获具体异常，并打印调试信息：

```python
try:
    result = self._call_llm(formatted_prompt)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON from LLM response: {e}")
    return None
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
    return None
```

## 核心数据结构

### AlphaFormula

```python
@dataclass
class AlphaFormula:
    name: str
    description: str
    formula_steps: List[Dict[str, Any]]  # [{"name": "ts_mean", "input": ["close"], "param": ["window"], "output": "v1"}]
    arguments: List[Dict[str, Any]]      # [{"window": 20}]
```

### AlphaNode（MCTS 节点）

```python
@dataclass
class AlphaNode:
    formula: AlphaFormula
    portrait: Dict[str, Any]
    parent: Optional['AlphaNode']
    children: List['AlphaNode']
    scores: Dict[str, float]             # 五维评分
    financial_metrics: Dict[str, float]  # 金融指标
    q_value: float                       # 综合评分均值
    visits: int                          # 访问次数
```

## 评估体系

### 五维评分（0-10 分）

1. **Effectiveness（有效性）**：基于 Rank IC 绝对值
2. **Stability（稳定性）**：基于 ICIR（信息比率）
3. **Turnover（换手率）**：基于因子自相关性，越低越好
4. **Diversity（多样性）**：与 Alpha101 和库内因子相关性，越低得分越高
5. **Overfitting Risk（过拟合风险）**：LLM Critic 评估

### 金融指标

- `rank_ic_mean`：平均 Rank IC
- `ic_mean`：平均普通 IC（Pearson）
- `icir`：IC 信息比率
- `turnover`：因子换手率
- `annualized_return`：年化收益率
- `sharpe_ratio`：夏普比率
- `max_drawdown`：最大回撤
- `excess_return`：超额收益
- `information_ratio`：信息比率（相对基准）

## 开发注意事项

### 1. LLM API 限制

- 设置 60 秒超时和 2 次自动重试
- 响应格式必须为 JSON，需处理 Markdown 代码块包裹的情况
- 中文提示词需确保 `ensure_ascii=False`

### 2. 算子约束

所有生成的公式必须严格使用 `AVAILABLE_OPERATORS` 中定义的算子：
- 时序算子：`ts_mean`, `ts_std`, `ts_rank`, `ts_corr`, `ts_delta`, `ts_sum`, `ts_min`, `ts_max`, `delay`, `ts_argmax`, `ts_argmin`, `decay_linear`
- 截面算子：`rank`, `scale`
- 数学算子：`log`, `abs`, `sign`, `add`, `subtract`, `multiply`, `divide`
- 其他：`sma`, `stddev`, `correlation`, `covariance`, `product`

### 3. 数据字段

可用字段仅限：`open`, `high`, `low`, `close`, `volume`, `vwap`

### 4. 结果导出

运行结果自动保存至：
- `result.txt`：文本格式完整记录
- `results/elite_factors_{mode}_{timestamp}.json`：JSON 格式
- `results/metrics_summary_{mode}_{timestamp}.csv`：CSV 格式

## 测试策略

项目目前采用**手动验证**方式：

1. **单元验证**：运行 `factor_calculator.py` 验证公式解析和计算
2. **集成测试**：运行 `main.py` 验证完整流程
3. **对比实验**：使用 `run_experiment.py` 进行多组对照实验
4. **可视化验证**：使用 `analyze_results.py` 生成图表检查分布

## 安全考虑

1. **API Key 管理**：`config.py` 中硬编码了 API Key，生产环境应改用环境变量
2. **数据路径**：数据文件路径硬编码为 Windows 格式（`D:/AAProject/Data`），跨平台需注意
3. **超时设置**：LLM 调用设置了 60 秒超时，避免无限阻塞

## 扩展开发

### 添加新的 Agent

1. 继承 `BaseAgent`
2. 实现 `execute(**kwargs)` 方法
3. 在 `inference/prompts/en/` 和 `inference/prompts/cn/` 中添加对应提示词文件

### 添加新的算子

1. 在 `factor_calculator.py` 的 `DEFAULT_FUNCS` 中实现算子逻辑
2. 在 `config.py` 的 `AVAILABLE_OPERATORS` 中添加名称
3. 更新 `OPERATOR_PARAM_COUNT` 和 `OPERATOR_INPUT_COUNT`

### 添加新的评估维度

1. 在 `EVALUATION_DIMENSIONS` 中添加维度名称
2. 在 `evaluator.py` 的 `simulate_evaluation` 中实现评分逻辑
