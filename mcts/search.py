import json
import os
import numpy as np
from config import PROMPT_DIR
from typing import Optional, List, Dict, Any
from utils.data_structures import AlphaNode, AlphaFormula
from alpha_library.library import AlphaLibrary
from evaluation.evaluator import simulate_evaluation, get_refinement_dimension
from agents.debate_agent import DebateAgent
from agents.synthesizer_agent import SynthesizerAgent
from agents.formula_agent import FormulaAgent
from config import (
    MCTS_EXPLORATION_WEIGHT, AVAILABLE_DATA_FIELDS, AVAILABLE_OPERATORS,
    OPERATOR_PARAM_COUNT, OPERATOR_INPUT_COUNT, SHOW_DEBATE_LOG
)

formula_agent = FormulaAgent(prompt_path=os.path.join(PROMPT_DIR, "formula_generation.txt"))
debate_agent = DebateAgent(prompt_path=os.path.join(PROMPT_DIR, "debate_turn.txt"))
synthesizer_agent = SynthesizerAgent(prompt_path=os.path.join(PROMPT_DIR, "debate_synthesis.txt"))


# 增强的校验函数
def is_formula_valid(formula: AlphaFormula) -> bool:
    """
    检查生成的AlphaFormula结构是否有效。
    会校验操作符、输入数量、参数数量、输入变量类型。
    """
    allowed_fields = set(AVAILABLE_DATA_FIELDS)
    allowed_ops = set(AVAILABLE_OPERATORS)
    defined_outputs = set()  # 存储过程中定义的变量名

    for step in formula.formula_steps:
        op_name = step.get("name")
        inputs = step.get("input", [])
        params = step.get("param", [])
        output_var = step.get("output")

        # 1. 校验操作符是否允许
        if op_name not in allowed_ops:
            print(f"校验失败: 使用了不允许的操作符 '{op_name}'")
            return False

        # 2. 校验参数数量 (param=[...])
        expected_param_count = OPERATOR_PARAM_COUNT.get(op_name)
        if expected_param_count is not None and len(params) != expected_param_count:
            print(f"校验失败: 操作符 '{op_name}' 应有 {expected_param_count} 个参数，但提供了 {len(params)} 个。")
            return False

        # 3. 校验输入数量 (input=[...])
        expected_input_count = OPERATOR_INPUT_COUNT.get(op_name)
        if expected_input_count is not None and len(inputs) != expected_input_count:
            print(f"校验失败: 操作符 '{op_name}' 应有 {expected_input_count} 个输入，但提供了 {len(inputs)} 个。")
            return False

        # 4. 校验输入变量是否有效
        for inp in inputs:
            if inp not in allowed_fields and inp not in defined_outputs:
                # 检查是否为常量数字
                try:
                    float(inp)  # 尝试将输入转换为数字
                    print(f"校验失败: 输入 '{inp}' 是一个常量数字，不允许出现在input列表中。")
                    return False
                except ValueError:
                    # 如果转换失败，说明它不是数字，是未定义变量
                    print(f"校验失败: 输入 '{inp}' 未定义或不是允许的基础字段")
                    return False

        defined_outputs.add(output_var)

    return True


class MCTS:
    def __init__(self, root: AlphaNode):
        self.root = root

    def _select_child(self, node: AlphaNode) -> Optional[AlphaNode]:
        if not node.children: return None
        return max(node.children, key=lambda child: child.calculate_uct(MCTS_EXPLORATION_WEIGHT))

    def select(self) -> AlphaNode:
        current_node = self.root
        while current_node.children:
            best_child = self._select_child(current_node)
            best_child_score = best_child.calculate_uct(MCTS_EXPLORATION_WEIGHT)
            virtual_self_visits = max(1, len(current_node.children))
            exploration_term_self = MCTS_EXPLORATION_WEIGHT * np.sqrt(np.log(current_node.visits) / virtual_self_visits)
            self_expansion_score = current_node.q_value + exploration_term_self
            if self_expansion_score > best_child_score:
                print(
                    f"决策: 扩展当前节点 '{current_node.portrait.get('name', '未命名')}' (自身分数 {self_expansion_score:.2f} > 最佳子节点分数 {best_child_score:.2f})")
                return current_node
            print(
                f"决策: 向下遍历至 '{best_child.portrait.get('name', '未命名')}' (子节点分数 {best_child_score:.2f} 更高)")
            current_node = best_child
        print(f"决策: 到达叶子节点，选择 '{current_node.portrait.get('name', '未命名')}' 进行扩展")
        return current_node

    # 确认 expand 方法使用了 SHOW_DEBATE_LOG
    def expand(self, node_to_expand: AlphaNode, freq_subtrees: List[str], alpha_repo: AlphaLibrary,
               ablation_mode: str = "debate") -> Optional[AlphaNode]:
        """
        扩展MCTS节点
        
        Args:
            node_to_expand: 待扩展的节点
            freq_subtrees: 频繁子树列表（FSA规避）
            alpha_repo: Alpha因子库
            ablation_mode: 扩展模式
                - "debate": 使用多智能体辩论（默认）
                - "refiner": 使用单智能体优化（Baseline）
        
        Returns:
            新创建的AlphaNode，如果扩展失败则返回None
        """
        print(f"\n--- 正在扩展节点: {node_to_expand.portrait.get('name', '未命名')} ---")
        print(f"--- FSA: 当前规避列表: {freq_subtrees} ---")
        print(f"--- 扩展模式: {ablation_mode} ---")

        refinement_dim = get_refinement_dimension(node_to_expand.scores)
        print(f"优化目标维度: {refinement_dim}")

        current_factor_type = node_to_expand.portrait.get('factor_type', '综合型')

        # --- 根据模式选择优化策略 ---
        if ablation_mode == "refiner":
            # Baseline模式：使用单智能体Refiner
            new_portrait = self._expand_by_refiner(node_to_expand, refinement_dim, 
                                                    current_factor_type, freq_subtrees)
        else:
            # 本方法模式：使用多智能体辩论
            new_portrait = self._expand_by_debate(node_to_expand, refinement_dim,
                                                  current_factor_type, freq_subtrees)

        if not new_portrait:
            return None

        new_portrait['factor_type'] = current_factor_type

        # --- 生成公式与评估 ---
        return self._generate_and_evaluate(node_to_expand, new_portrait, refinement_dim, alpha_repo)

    def _expand_by_refiner(self, node_to_expand: AlphaNode, refinement_dim: str,
                           factor_type: str, freq_subtrees: List[str]) -> Optional[Dict[str, Any]]:
        """使用Refiner Agent进行单智能体优化（Baseline模式）"""
        from agents.refiner_agent import RefinerAgent

        print("\n--- [Baseline] 使用Refiner Agent进行优化 ---")

        refiner_agent = RefinerAgent(
            prompt_path=os.path.join(PROMPT_DIR, "alpha_refinement.txt")
        )

        # 构建优化建议
        suggestions = f"请重点优化因子的{refinement_dim}维度，保持{factor_type}类型特征。"
        if freq_subtrees:
            suggestions += f" 避免使用以下常见结构: {freq_subtrees}"

        # 调用Refiner Agent
        new_portrait = refiner_agent.execute(
            original_formula=node_to_expand.formula,
            original_portrait=node_to_expand.portrait,
            suggestions=suggestions,
            freq_subtrees=freq_subtrees
        )

        if not new_portrait:
            print("[Baseline] Refiner未能生成新画像，跳过本次扩展。")
            return None

        return new_portrait

    def _expand_by_debate(self, node_to_expand: AlphaNode, refinement_dim: str,
                          factor_type: str, freq_subtrees: List[str]) -> Optional[Dict[str, Any]]:
        """使用多智能体辩论进行优化（本研究方法）"""
        from config import DEBATE_ROUNDS

        print("\n--- 开始因子优化辩论 ---")

        debate_history = ""
        num_rounds = DEBATE_ROUNDS

        personas = {
            "Agent_A": f"你是一名积极的量化研究员(主Agent)，专注于最大化Alpha因子的 '{refinement_dim}' 表现，倾向于探索更有效的复杂结构。请针对 '{factor_type}' 类型进行思考。",
            "Agent_B": f"你是一名谨慎的量化研究员(主Agent)，在提升 '{refinement_dim}' 的同时，高度关注简洁性、低换手率和过拟合风险。请确保优化后的因子仍然符合 '{factor_type}' 类型。",
            "Critic": f"你是一名批判性的评审员(Critic Agent)，负责审视主Agent提出的优化建议。质疑其合理性、潜在风险（过拟合、换手率、偏离因子类型）、是否有效解决了 '{refinement_dim}' 问题，以及是否忽略了FSA规避列表 {freq_subtrees}。"
        }
        agent_ids = ["Agent_A", "Agent_B", "Critic"]

        print("\n--- 开始因子优化辩论 ---")
        for round_num in range(1, num_rounds + 1):
            if SHOW_DEBATE_LOG:
                print(f"\n--- 第 {round_num} 轮辩论 ---")

            for agent_id in agent_ids:
                if SHOW_DEBATE_LOG:
                    print(f"\n{agent_id} 发言:")

                persona = personas[agent_id]

                speech = debate_agent.execute(
                    persona=persona,
                    current_alpha_portrait=node_to_expand.portrait,
                    optimization_target=refinement_dim,
                    factor_type=factor_type,
                    fsa_avoid_list=freq_subtrees,
                    debate_history=debate_history
                )

                if speech:
                    analysis = speech.get("analysis", "无分析")
                    contribution = speech.get("contribution", "无贡献")

                    if SHOW_DEBATE_LOG:
                        print(f"  分析: {analysis}")
                        print(f"  贡献: {contribution}")

                    debate_history += f"{agent_id}:\nAnalysis: {analysis}\nContribution: {contribution}\n\n"
                else:
                    if SHOW_DEBATE_LOG:
                        print(f"  {agent_id} 发言失败，跳过。")

        if SHOW_DEBATE_LOG:
            print("\n--- 辩论结束 ---")

        # --- 结果合成 ---
        print("\n--- 正在合成最终优化画像 ---")
        new_portrait = synthesizer_agent.execute(
            original_alpha_portrait=node_to_expand.portrait,
            optimization_target=refinement_dim,
            factor_type=factor_type,
            fsa_avoid_list=freq_subtrees,
            debate_history=debate_history
        )

        if not new_portrait:
            print("合成智能体未能生成最终画像，跳过本次扩展。")
            return None

        return new_portrait

    def _generate_and_evaluate(self, parent_node: AlphaNode, new_portrait: Dict[str, Any],
                               refinement_dim: str, alpha_repo: AlphaLibrary) -> Optional[AlphaNode]:
        """生成公式并评估新节点"""
        # --- 生成公式 ---
        print("--- 正在生成新公式 ---")
        new_formula = formula_agent.execute(alpha_portrait=new_portrait)
        if not new_formula:
            print("公式智能体未能合成公式，跳过本次扩展。")
            return None

        # 使用增强后的校验函数
        if not is_formula_valid(new_formula):
            print("新生成的公式结构校验失败，跳过本次扩展。")
            return None

        # 创建新节点
        new_node = AlphaNode(
            formula=new_formula,
            portrait=new_portrait,
            parent=parent_node,
            refinement_summary=f"经优化(目标:{refinement_dim})。新思路: {new_portrait.get('description', '无')}"
        )

        # 评估新节点
        print("--- 正在评估新因子 ---")
        new_scores = simulate_evaluation(new_formula, new_node, alpha_repo)
        new_node.scores = new_scores
        new_node.q_value = np.mean(list(new_scores.values())) if new_scores else 0.0

        parent_node.children.append(new_node)
        print(f"\n已创建新节点: '{new_node.portrait.get('name', '未命名')}', Q值为: {new_node.q_value:.2f}")

        print("--- 新节点详细信息 ---")
        formula_str = new_node.formula.to_expression_string()
        print(f"  因子公式: {formula_str}")
        scores_str = json.dumps(new_node.scores, indent=4)
        print(f"  五维得分:\n{scores_str}")
        # 打印新金融指标
        metrics_str = json.dumps(new_node.financial_metrics, indent=4, default=lambda x: f"{x:.4f}")
        print(f"  金融指标:\n{metrics_str}")
        print("--------------------")

        return new_node

    def backpropagate(self, node: AlphaNode):
        current_node = node
        new_score = node.q_value
        while current_node is not None:
            current_node.visits += 1
            if new_score > current_node.q_value:
                current_node.q_value = new_score
            print(
                f"反向传播至: '{current_node.portrait.get('name', '未命名')}', 新访问次数: {current_node.visits}, 更新后Q值: {current_node.q_value:.2f}")
            current_node = current_node.parent