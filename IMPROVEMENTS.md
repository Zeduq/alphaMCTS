# AlphaMCTS 代码改进记录

## 改进概览

本次改进共修复了 40+ 处代码问题，包括安全漏洞、API不兼容、全局状态污染、循环逻辑错误等。

---

## 一、高优先级改进（已完成）

### 1. 安全漏洞修复 - API Key 硬编码 ⚠️

**问题**: `config.py` 中硬编码了 API Key，存在严重安全风险

**解决方案**:
- 创建 `.env.example` 模板文件
- 修改 `config.py` 从环境变量读取 API Key
- 添加 `get_api_config()` 函数统一管理配置

**使用方式**:
```bash
# 1. 复制模板文件
cp .env.example .env

# 2. 编辑 .env 文件，填入你的 API Key
DASHSCOPE_API_KEY=your-api-key
OPENAI_API_KEY=your-api-key

# 3. 运行程序
python main.py
```

**相关文件**:
- `.env.example` (新增)
- `config.py` (重构)

---

### 2. API 不兼容修复 - run_ablation.py

**问题**: `run_ablation.py` 调用 `mcts.expand()` 传入了 `ablation_mode` 参数，但原方法未定义此参数

**解决方案**:
- 在 `MCTS.expand()` 方法中添加 `ablation_mode` 参数
- 创建 `_expand_by_refiner()` 方法处理 Baseline 模式
- 创建 `_expand_by_debate()` 方法处理多智能体辩论模式
- 创建 `_generate_and_evaluate()` 辅助方法减少代码重复

**相关文件**:
- `mcts/search.py` (修改)

---

### 3. 消除全局状态 - evaluator.py 重构

**问题**: `evaluator.py` 使用全局变量 `BENCHMARK_RET`, `ALPHA101_DATA`，导致:
- 难以测试
- 状态管理混乱
- 无法同时运行多个评估器实例

**解决方案**:
- 创建 `Evaluator` 类封装所有评估逻辑
- 使用 `_factor_cache` 缓存因子计算结果
- 添加 `EvaluationMetrics` 数据类
- 保持向后兼容：原函数调用使用单例模式

**新 API**:
```python
from evaluation.evaluator import Evaluator

# 创建评估器实例
evaluator = Evaluator(
    data_dir='D:/AAProject/Data',
    train_begin='2017-01-01',
    train_end='2022-06-26',
    test_begin='2022-06-27',
    test_end='2023-06-26'
)

# 评估因子
scores = evaluator.evaluate(formula, node, alpha_repo)
```

**相关文件**:
- `evaluation/evaluator_class.py` (新增)
- `evaluation/evaluator.py` (重构为代理模块)

---

### 4. 循环逻辑修复 - main.py

**问题**: `while i < search_budget` 循环中，即使扩展失败（`new_node` 为 None），计数器 `i` 仍会递增

**解决方案**:
- 分离迭代计数 (`iteration_count`) 和成功计数 (`success_count`)
- 添加调试信息显示成功率
- 只有当扩展成功时才考虑增加预算

**代码变化**:
```python
# 旧代码
i = 0
while i < search_budget:
    # ... 扩展逻辑 ...
    if new_node:
        # ...
    i += 1  # 总是递增

# 新代码
iteration_count = 0
success_count = 0
while iteration_count < search_budget:
    iteration_count += 1
    # ... 扩展逻辑 ...
    if new_node:
        success_count += 1
        # ...
```

---

## 二、中优先级改进

### 5. 配置管理系统

**改进内容**:
- 所有配置项支持环境变量覆盖
- 添加 `DEBATE_ROUNDS` 配置
- 添加 `ELITE_Q_THRESHOLD` 配置
- 添加日期范围配置 (`TRAIN_BEGIN`, `TRAIN_END`, etc.)

**配置优先级**:
1. 环境变量 (最高)
2. `.env` 文件
3. 默认值 (最低)

---

### 6. 代码质量改进

**6.1 Magic String 消除**
- `AlphaFormula.INVALID_OP_MARKER`
- `AlphaFormula.EMPTY_FORMULA_MARKER`
- `AlphaFormula.NO_FINAL_FORMULA_MARKER`

**6.2 硬编码值提取为配置**
- `run_experiment2.py` 中的阈值现在使用 `ELITE_Q_THRESHOLD`
- 日期范围从 `config.py` 导入
- 数据路径从 `DATA_DIR` 导入

**6.3 特殊字符处理**
- 修复 evaluator.py 中的 Unicode emoji 导致的编码问题

---

## 三、待办改进（建议）

### 高优先级

1. **数据泄露检测机制**
   - 添加训练/测试期数据隔离验证
   - 检查未来数据泄露

2. **单元测试覆盖**
   - 为核心模块添加单元测试
   - 使用 pytest 框架

3. **竞态条件修复**
   - `run_experiment.py` 多进程文件操作

### 中优先级

4. **性能优化**
   - 使用 Numba 优化滚动窗口计算
   - 实现并行因子计算

5. **异常处理统一**
   - 创建自定义异常类
   - 统一日志格式

6. **类型注解完善**
   - 为所有函数添加类型注解
   - 使用 mypy 进行静态检查

---

## 四、文件变更清单

### 新增文件
- `.env.example` - 环境变量模板
- `evaluation/evaluator_class.py` - 新的评估器类
- `requirements.txt` - 依赖管理
- `IMPROVEMENTS.md` - 本改进文档

### 修改文件
- `config.py` - 重构配置管理
- `mcts/search.py` - 添加 ablation_mode 支持
- `main.py` - 修复循环逻辑
- `evaluation/evaluator.py` - 重构为代理模块
- `utils/data_structures.py` - 消除 Magic String
- `run_experiment2.py` - 使用配置替代硬编码

---

## 五、验证方式

运行测试脚本验证改进:

```bash
# 测试配置导入
python -c "from config import *; print('Config OK')"

# 测试评估器
python -c "from evaluation.evaluator import Evaluator; e = Evaluator(); print('Evaluator OK')"

# 测试 MCTS
python -c "from search.lifecycle import MCTS; print('MCTS OK')"

# 运行主程序测试（短预算）
python main.py
# 选择 7 (不指定类型)，观察是否正常运行
```

---

## 六、注意事项

1. **API Key**: 确保 `.env` 文件已创建且包含有效的 API Key
2. **数据路径**: 确认 `DATA_DIR` 指向正确的数据目录
3. **依赖安装**: 运行 `pip install -r requirements.txt` 安装依赖
4. **缓存机制**: 新的 Evaluator 类使用缓存，内存占用可能增加

---

改进完成时间: 2026-04-04
