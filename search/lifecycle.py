import json
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import PROMPT_DIR
from typing import Optional, List, Dict, Any
from search.tree import AlphaNode, AlphaFormula
from constraint.library import AlphaLibrary
from evaluation.evaluator import simulate_evaluation, get_refinement_dimension
from config import (
    MCTS_EXPLORATION_WEIGHT, AVAILABLE_DATA_FIELDS, AVAILABLE_OPERATORS,
    OPERATOR_PARAM_COUNT, OPERATOR_INPUT_COUNT, SHOW_DEBATE_LOG
)


def is_formula_valid(formula: AlphaFormula) -> bool:
    allowed_fields = set(AVAILABLE_DATA_FIELDS)
    allowed_ops = set(AVAILABLE_OPERATORS)
    defined_outputs = set()

    for step in formula.formula_steps:
        op_name = step.get("name")
        inputs = step.get("input", [])
        params = step.get("param", [])
        output_var = step.get("output")

        if op_name not in allowed_ops:
            print(f"校验失败: 使用了不允许的操作符 '{op_name}'")
            return False

        expected_param_count = OPERATOR_PARAM_COUNT.get(op_name)
        if expected_param_count is not None and len(params) != expected_param_count:
            print(f"校验失败: 操作符 '{op_name}' 应有 {expected_param_count} 个参数，但提供了 {len(params)} 个。")
            return False

        expected_input_count = OPERATOR_INPUT_COUNT.get(op_name)
        if expected_input_count is not None and len(inputs) != expected_input_count:
            print(f"校验失败: 操作符 '{op_name}' 应有 {expected_input_count} 个输入，但提供了 {len(inputs)} 个。")
            return False

        for inp in inputs:
            if inp not in allowed_fields and inp not in defined_outputs:
                try:
                    float(inp)
                    print(f"校验失败: 输入 '{inp}' 是一个常量数字，不允许出现在input列表中。")
                    return False
                except ValueError:
                    print(f"校验失败: 输入 '{inp}' 未定义或不是允许的基础字段")
                    return False

        defined_outputs.add(output_var)

    return True


class MCTS:
    def __init__(self, root: AlphaNode):
        self.root = root
        from inference.agents.formula import FormulaAgent
        from inference.agents.debate import DebateAgent
        from inference.agents.synthesizer import SynthesizerAgent
        self.formula_agent = FormulaAgent(prompt_path=os.path.join(PROMPT_DIR, "formula_generation.txt"))
        self.debate_agent = DebateAgent(prompt_path=os.path.join(PROMPT_DIR, "debate_turn.txt"))
        self.synthesizer_agent = SynthesizerAgent(prompt_path=os.path.join(PROMPT_DIR, "debate_synthesis.txt"))

    def _select_child(self, node: AlphaNode) -> Optional[AlphaNode]:
        if not node.children:
            return None
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
                return current_node
            current_node = best_child
        return current_node

    def expand(self, node_to_expand: AlphaNode, freq_subtrees: List[str], alpha_repo: AlphaLibrary,
               ablation_mode: str = "debate") -> Optional[AlphaNode]:
        refinement_dim = get_refinement_dimension(node_to_expand.scores)
        current_factor_type = node_to_expand.portrait.get('factor_type', '综合型')

        if ablation_mode == "refiner":
            new_portrait = self._expand_by_refiner(node_to_expand, refinement_dim,
                                                    current_factor_type, freq_subtrees)
        else:
            new_portrait = self._expand_by_debate(node_to_expand, refinement_dim,
                                                  current_factor_type, freq_subtrees)

        if not new_portrait:
            return None

        new_portrait['factor_type'] = current_factor_type
        return self._generate_and_evaluate(node_to_expand, new_portrait, refinement_dim, alpha_repo)

    def _expand_by_refiner(self, node_to_expand: AlphaNode, refinement_dim: str,
                           factor_type: str, freq_subtrees: List[str]) -> Optional[Dict[str, Any]]:
        from inference.agents.refiner import RefinerAgent

        refiner_agent = RefinerAgent(
            prompt_path=os.path.join(PROMPT_DIR, "alpha_refinement.txt")
        )

        suggestions = f"请重点优化因子的{refinement_dim}维度，保持{factor_type}类型特征。"
        if freq_subtrees:
            suggestions += f" 避免使用以下常见结构: {freq_subtrees}"

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
        from config import DEBATE_ROUNDS

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

            def _agent_speech(agent_id: str):
                persona = personas[agent_id]
                speech = self.debate_agent.execute(
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
                        print(f"\n{agent_id} 发言:\n  分析: {analysis}\n  贡献: {contribution}")
                    return agent_id, analysis, contribution
                else:
                    if SHOW_DEBATE_LOG:
                        print(f"\n{agent_id} 发言失败，跳过。")
                    return agent_id, None, None

            round_results = {}
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(_agent_speech, aid): aid for aid in agent_ids}
                for future in as_completed(futures):
                    agent_id, analysis, contribution = future.result()
                    round_results[agent_id] = (analysis, contribution)

            for agent_id in agent_ids:
                analysis, contribution = round_results[agent_id]
                if analysis is not None:
                    debate_history += f"{agent_id}:\nAnalysis: {analysis}\nContribution: {contribution}\n\n"

        if SHOW_DEBATE_LOG:
            print("\n--- 辩论结束 ---")

        new_portrait = self.synthesizer_agent.execute(
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
        new_formula = self.formula_agent.execute(alpha_portrait=new_portrait)
        if not new_formula:
            print("公式智能体未能合成公式，跳过本次扩展。")
            return None

        if not is_formula_valid(new_formula):
            print("新生成的公式结构校验失败，跳过本次扩展。")
            return None

        new_node = AlphaNode(
            formula=new_formula,
            portrait=new_portrait,
            parent=parent_node,
            refinement_summary=f"经优化(目标:{refinement_dim})。新思路: {new_portrait.get('description', '无')}"
        )

        new_scores = simulate_evaluation(new_formula, new_node, alpha_repo)
        new_node.scores = new_scores
        new_node.q_value = np.mean(list(new_scores.values())) if new_scores else 0.0

        parent_node.children.append(new_node)
        print(f"[扩展成功] {new_node.portrait.get('name', '未命名')} | Q={new_node.q_value:.2f} | 目标={refinement_dim}")
        return new_node

    def backpropagate(self, node: AlphaNode):
        current_node = node
        new_score = node.q_value
        while current_node is not None:
            current_node.visits += 1
            if new_score > current_node.q_value:
                current_node.q_value = new_score
            current_node = current_node.parent
