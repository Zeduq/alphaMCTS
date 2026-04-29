"""
AlphaMCTS 可视化 GUI 系统

功能：
1. 左侧：交互式 MCTS 搜索树可视化
   - 实时展示搜索树构建过程
   - 节点悬停显示详细信息（五维评分、因子画像、公式、Rank IC等）
   - 支持缩放和平移

2. 右侧：模拟终端
   - 实时显示运行日志
   - 支持命令输入

使用方法：
    python mcts_gui.py
"""

import sys
import os
import json
import math
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLabel, QPushButton, QComboBox, QSpinBox, QGraphicsView,
    QGraphicsScene, QGraphicsItem, QGraphicsEllipseItem, QGraphicsLineItem,
    QGraphicsTextItem, QGraphicsProxyWidget, QToolTip, QMessageBox,
    QSplitter, QFrame, QGroupBox, QFormLayout, QProgressBar, QFileDialog,
    QTabWidget, QTreeWidget, QTreeWidgetItem, QHeaderView
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QObject, QPointF, QRectF, QTimer
)
from PyQt5.QtGui import (
    QBrush, QPen, QColor, QFont, QPainter, QFontMetrics, QPalette,
    QLinearGradient, QRadialGradient
)

# 导入项目模块
from config import PROMPT_DIR, INITIAL_SEARCH_BUDGET, EFFECTIVENESS_THRESHOLD
from mcts.search import MCTS
from utils.data_structures import AlphaNode, AlphaFormula
from utils.exporter import export_elite_factors
from alpha_library.library import AlphaLibrary
from agents.portrait_agent import PortraitAgent
from agents.formula_agent import FormulaAgent
from evaluation.evaluator import simulate_evaluation
from fsa.fsa_miner import mine_frequent_subtrees


# =============================================================================
# MCTS 后台工作线程
# =============================================================================

class MCTSWorker(QObject):
    """
    MCTS搜索工作线程
    
    在后台线程执行搜索，避免阻塞GUI主线程
    """
    # 信号定义
    root_node_ready = pyqtSignal(AlphaNode)  # 根节点初始化完成
    iteration_started = pyqtSignal(int, int)  # 当前迭代, 总预算
    node_expanded = pyqtSignal(AlphaNode, AlphaNode)  # 父节点, 新节点
    node_evaluated = pyqtSignal(AlphaNode)  # 节点评估完成
    backprop_done = pyqtSignal(AlphaNode)  # 反向传播完成
    search_finished = pyqtSignal()  # 搜索完成
    search_error = pyqtSignal(str)  # 搜索错误
    log_message = pyqtSignal(str, str)  # 日志消息, 类型
    
    def __init__(self):
        super().__init__()
        self.mcts: Optional[MCTS] = None
        self.alpha_repo = AlphaLibrary()
        self.is_running = False
        self.stop_requested = False
    
    def initialize_root(self, factor_type: str):
        """初始化根节点"""
        try:
            self.log_message.emit("正在初始化根节点...", "info")
            
            portrait_agent = PortraitAgent(prompt_path=os.path.join(PROMPT_DIR, "portrait_generation.txt"))
            formula_agent = FormulaAgent(prompt_path=os.path.join(PROMPT_DIR, "formula_generation.txt"))
            
            root_portrait = portrait_agent.execute(freq_subtrees=[], factor_type=factor_type)
            if not root_portrait:
                raise Exception("生成初始Alpha画像失败")
            
            root_formula = formula_agent.execute(alpha_portrait=root_portrait)
            if not root_formula:
                raise Exception("合成初始Alpha公式失败")
            
            root_node = AlphaNode(formula=root_formula, portrait=root_portrait)
            root_scores = simulate_evaluation(root_node.formula, root_node, AlphaLibrary())
            root_node.scores = root_scores
            root_node.q_value = np.mean(list(root_scores.values())) if root_scores else 0
            
            self.mcts = MCTS(root=root_node)
            self.alpha_repo = AlphaLibrary()
            
            self.root_node_ready.emit(root_node)
            self.log_message.emit(f"根节点初始化完成! Q值={root_node.q_value:.2f}", "success")
            
        except Exception as e:
            self.search_error.emit(f"初始化失败: {str(e)}")
    
    def run_search(self, budget: int, threshold: int):
        """运行搜索循环"""
        if not self.mcts:
            self.search_error.emit("MCTS未初始化")
            return
        
        self.is_running = True
        self.stop_requested = False
        
        try:
            for iteration in range(1, budget + 1):
                if self.stop_requested:
                    self.log_message.emit("搜索被用户停止", "warning")
                    break
                
                self.iteration_started.emit(iteration, budget)
                self.log_message.emit(f"第 {iteration}/{budget} 次迭代", "debug")
                
                # 挖掘频繁子树
                freq_subtrees = mine_frequent_subtrees(self.alpha_repo.alphas, top_k=3)
                
                # 选择节点
                node_to_expand = self.mcts.select()
                
                # 扩展节点
                new_node = self.mcts.expand(node_to_expand, freq_subtrees, self.alpha_repo)
                
                if new_node:
                    self.node_expanded.emit(node_to_expand, new_node)
                    
                    # 反向传播
                    self.mcts.backpropagate(new_node)
                    self.backprop_done.emit(new_node)
                    
                    # 判断入库
                    if new_node.scores.get("Effectiveness", 0) >= threshold:
                        self.alpha_repo.add(new_node)
                        self.log_message.emit(f"✓ 新节点入库! Q值={new_node.q_value:.2f}", "success")
                
                # 小延迟让GUI有时间更新
                QThread.msleep(100)
            
            self.search_finished.emit()
            
        except Exception as e:
            import traceback
            self.search_error.emit(f"搜索错误: {str(e)}\n{traceback.format_exc()}")
        finally:
            self.is_running = False
    
    def stop(self):
        """请求停止搜索"""
        self.stop_requested = True
        self.log_message.emit("正在停止搜索...", "warning")


# =============================================================================
# 树节点图形项
# =============================================================================

class TreeNodeItem(QGraphicsEllipseItem):
    """MCTS树节点的可视化图形项"""
    
    def __init__(self, node: AlphaNode, x: float, y: float, radius: float = 25):
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.node = node
        self.radius = radius
        self.setPos(x, y)
        
        # 设置可交互
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, False)
        
        # 根据Q值设置颜色
        self.update_color()
        
        # 节点标签
        self.label = None
        self.create_label()
    
    def update_color(self):
        """根据节点Q值更新颜色"""
        q_value = self.node.q_value if hasattr(self.node, 'q_value') else 0
        
        # Q值映射到颜色：低(红色) -> 高(绿色)
        if q_value < 3:
            color = QColor(231, 76, 60)  # 红色
        elif q_value < 5:
            color = QColor(241, 196, 15)  # 黄色
        elif q_value < 7:
            color = QColor(46, 204, 113)  # 浅绿
        else:
            color = QColor(39, 174, 96)  # 深绿
        
        self.setBrush(QBrush(color))
        self.setPen(QPen(QColor(44, 62, 80), 2))
    
    def create_label(self):
        """创建节点标签"""
        name = self.node.portrait.get('name', 'Unknown')[:8]
        self.label = QGraphicsTextItem(name, self)
        self.label.setFont(QFont("Microsoft YaHei", 8))
        self.label.setDefaultTextColor(QColor(44, 62, 80))
        
        # 居中显示
        text_rect = self.label.boundingRect()
        self.label.setPos(-text_rect.width() / 2, -text_rect.height() / 2)
    
    def hoverEnterEvent(self, event):
        """鼠标悬停进入事件"""
        self.setPen(QPen(QColor(52, 152, 219), 4))
        self.show_tooltip()
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        """鼠标悬停离开事件"""
        self.setPen(QPen(QColor(44, 62, 80), 2))
        super().hoverLeaveEvent(event)
    
    def show_tooltip(self):
        """显示节点详细信息"""
        # 构建提示信息
        tooltip_text = self.build_tooltip_text()
        QToolTip.showText(
            self.scene().views()[0].mapToGlobal(
                self.scene().views()[0].mapFromScene(self.scenePos())
            ),
            tooltip_text,
            self.scene().views()[0]
        )
    
    def build_tooltip_text(self) -> str:
        """构建节点详细信息文本"""
        lines = []
        lines.append("<h3>🎯 节点详情</h3>")
        
        # 基本信息
        name = self.node.portrait.get('name', '未命名')
        desc = self.node.portrait.get('description', '无描述')
        lines.append(f"<b>名称:</b> {name}")
        lines.append(f"<b>Q值:</b> {self.node.q_value:.2f}")
        lines.append(f"<b>访问次数:</b> {self.node.visits}")
        lines.append("")
        
        # 五维评分
        if hasattr(self.node, 'scores') and self.node.scores:
            lines.append("<b>📊 五维评分:</b>")
            for dim, score in self.node.scores.items():
                lines.append(f"  • {dim}: {score:.2f}")
            lines.append("")
        
        # 金融指标
        if hasattr(self.node, 'financial_metrics') and self.node.financial_metrics:
            fm = self.node.financial_metrics
            lines.append("<b>💰 金融指标:</b>")
            lines.append(f"  • Rank IC: {fm.get('rank_ic_mean', 0):.4f}")
            lines.append(f"  • ICIR: {fm.get('icir', 0):.4f}")
            lines.append(f"  • Sharpe: {fm.get('sharpe_ratio', 0):.4f}")
            lines.append(f"  • 换手率: {fm.get('turnover', 0):.4f}")
            lines.append("")
        
        # 因子公式
        if hasattr(self.node, 'formula') and self.node.formula:
            formula_str = self.node.formula.to_expression_string()
            lines.append("<b>📝 因子公式:</b>")
            lines.append(f"<code>{formula_str[:100]}{'...' if len(formula_str) > 100 else ''}</code>")
        
        return "<br>".join(lines)


class TreeEdgeItem(QGraphicsLineItem):
    """树边的可视化图形项"""
    
    def __init__(self, parent_node: TreeNodeItem, child_node: TreeNodeItem):
        # 计算起点和终点
        parent_pos = parent_node.scenePos()
        child_pos = child_node.scenePos()
        
        super().__init__(parent_pos.x(), parent_pos.y(), child_pos.x(), child_pos.y())
        
        # 设置线条样式
        self.setPen(QPen(QColor(149, 165, 166), 1.5))
        
        self.parent_node = parent_node
        self.child_node = child_node
    
    def update_position(self):
        """更新边的位置（当节点移动时）"""
        parent_pos = self.parent_node.scenePos()
        child_pos = self.child_node.scenePos()
        self.setLine(parent_pos.x(), parent_pos.y(), child_pos.x(), child_pos.y())


# =============================================================================
# 搜索树场景
# =============================================================================

class MCTSTreeScene(QGraphicsScene):
    """MCTS搜索树场景"""
    
    node_clicked = pyqtSignal(AlphaNode)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSceneRect(-2000, -100, 4000, 2000)
        self.node_items: Dict[int, TreeNodeItem] = {}
        self.edge_items: list = []
        self.level_height = 120  # 层级高度
        self.node_spacing = 100  # 节点间距
    
    def clear_tree(self):
        """清空树"""
        self.clear()
        self.node_items.clear()
        self.edge_items.clear()
    
    def add_node(self, node: AlphaNode, parent_item: Optional[TreeNodeItem] = None):
        """添加节点到场景"""
        # 计算节点位置
        x, y = self.calculate_node_position(node, parent_item)
        
        # 创建图形项
        node_item = TreeNodeItem(node, x, y)
        self.addItem(node_item)
        self.node_items[id(node)] = node_item
        
        # 创建边
        if parent_item:
            edge = TreeEdgeItem(parent_item, node_item)
            self.addItem(edge)
            self.edge_items.append(edge)
            # 确保边在节点后面
            edge.setZValue(-1)
        
        return node_item
    
    def calculate_node_position(self, node: AlphaNode, parent_item: Optional[TreeNodeItem]) -> tuple:
        """计算节点位置"""
        if parent_item is None:
            # 根节点居中
            return 0, 50
        
        # 根据深度和兄弟节点数量计算位置
        depth = self.get_node_depth(node)
        y = depth * self.level_height
        
        # 简单的水平布局
        parent_x = parent_item.x()
        sibling_count = len([c for c in self.node_items.values() 
                            if abs(c.y() - y) < 10])  # 同层级节点数
        
        # 在父节点两侧分布
        if sibling_count % 2 == 0:
            x = parent_x + (sibling_count + 1) * self.node_spacing / 2
        else:
            x = parent_x - (sibling_count + 1) * self.node_spacing / 2
        
        return x, y
    
    def get_node_depth(self, node: AlphaNode) -> int:
        """获取节点深度"""
        depth = 0
        current = node
        while current.parent:
            depth += 1
            current = current.parent
        return depth
    
    def update_tree_layout(self):
        """更新树的布局（重新计算位置）"""
        # 这里可以实现更复杂的树布局算法
        pass


# =============================================================================
# 终端模拟器
# =============================================================================

class TerminalWidget(QTextEdit):
    """模拟终端的文本显示组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.WidgetWidth)
        
        # 设置终端样式
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                border: 1px solid #333;
                padding: 10px;
            }
        """)
        
        # 最大行数限制
        self.max_lines = 1000
    
    def append_log(self, text: str, log_type: str = "info"):
        """添加日志"""
        # 根据类型设置颜色
        colors = {
            "info": "#d4d4d4",
            "success": "#4ec9b0",
            "warning": "#dcdcaa",
            "error": "#f44747",
            "debug": "#808080"
        }
        color = colors.get(log_type, "#d4d4d4")
        
        # 添加时间戳
        timestamp = datetime.now().strftime("%H:%M:%S")
        html = f'<span style="color: #858585;">[{timestamp}]</span> <span style="color: {color};">{text}</span>'
        
        self.append(html)
        
        # 限制行数
        self.limit_lines()
        
        # 滚动到底部
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def limit_lines(self):
        """限制最大行数"""
        doc = self.document()
        if doc.blockCount() > self.max_lines:
            cursor = self.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()


# =============================================================================
# 节点详情面板
# =============================================================================

class NodeDetailPanel(QGroupBox):
    """节点详细信息面板"""
    
    def __init__(self, parent=None):
        super().__init__("节点详情", parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QFormLayout(self)
        layout.setSpacing(10)
        
        # 基本信息
        self.name_label = QLabel("-")
        self.q_value_label = QLabel("-")
        self.visits_label = QLabel("-")
        
        layout.addRow("<b>名称:</b>", self.name_label)
        layout.addRow("<b>Q值:</b>", self.q_value_label)
        layout.addRow("<b>访问次数:</b>", self.visits_label)
        
        # 五维评分
        self.scores_tree = QTreeWidget()
        self.scores_tree.setHeaderLabels(["维度", "评分"])
        self.scores_tree.setColumnWidth(0, 100)
        self.scores_tree.setMaximumHeight(150)
        layout.addRow("<b>五维评分:</b>", self.scores_tree)
        
        # 金融指标
        self.metrics_tree = QTreeWidget()
        self.metrics_tree.setHeaderLabels(["指标", "数值"])
        self.metrics_tree.setColumnWidth(0, 100)
        self.metrics_tree.setMaximumHeight(150)
        layout.addRow("<b>金融指标:</b>", self.metrics_tree)
        
        # 因子公式
        self.formula_text = QTextEdit()
        self.formula_text.setReadOnly(True)
        self.formula_text.setMaximumHeight(100)
        self.formula_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        layout.addRow("<b>因子公式:</b>", self.formula_text)
        
        # 描述
        self.desc_label = QLabel("-")
        self.desc_label.setWordWrap(True)
        layout.addRow("<b>描述:</b>", self.desc_label)
    
    def update_node(self, node: AlphaNode):
        """更新显示的节点信息"""
        if not node:
            return
        
        # 基本信息
        self.name_label.setText(node.portrait.get('name', '未命名'))
        self.q_value_label.setText(f"{node.q_value:.2f}")
        self.visits_label.setText(str(node.visits))
        
        # 五维评分
        self.scores_tree.clear()
        if hasattr(node, 'scores') and node.scores:
            for dim, score in node.scores.items():
                item = QTreeWidgetItem([dim, f"{score:.2f}"])
                # 根据分数设置颜色
                if score >= 7:
                    item.setForeground(1, QColor(39, 174, 96))
                elif score >= 4:
                    item.setForeground(1, QColor(241, 196, 15))
                else:
                    item.setForeground(1, QColor(231, 76, 60))
                self.scores_tree.addTopLevelItem(item)
        
        # 金融指标
        self.metrics_tree.clear()
        if hasattr(node, 'financial_metrics') and node.financial_metrics:
            fm = node.financial_metrics
            metrics = [
                ("Rank IC", f"{fm.get('rank_ic_mean', 0):.4f}"),
                ("ICIR", f"{fm.get('icir', 0):.4f}"),
                ("Sharpe", f"{fm.get('sharpe_ratio', 0):.4f}"),
                ("换手率", f"{fm.get('turnover', 0):.4f}"),
                ("年化收益", f"{fm.get('annualized_return', 0):.4f}"),
                ("最大回撤", f"{fm.get('max_drawdown', 0):.4f}"),
            ]
            for name, value in metrics:
                item = QTreeWidgetItem([name, value])
                self.metrics_tree.addTopLevelItem(item)
        
        # 因子公式
        if hasattr(node, 'formula') and node.formula:
            formula_str = node.formula.to_expression_string()
            self.formula_text.setText(formula_str)
        else:
            self.formula_text.setText("无公式")
        
        # 描述
        self.desc_label.setText(node.portrait.get('description', '无描述'))


# =============================================================================
# 控制面板
# =============================================================================

class ControlPanel(QGroupBox):
    """控制面板"""
    
    start_search = pyqtSignal(str, int, int)  # factor_type, budget, threshold
    stop_search = pyqtSignal()
    clear_tree = pyqtSignal()
    export_results = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__("控制面板", parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 因子类型选择
        form_layout = QFormLayout()
        
        self.factor_type_combo = QComboBox()
        self.factor_type_combo.addItems([
            "动量因子", "波动率因子", "情绪/另类因子",
            "价值因子", "质量因子", "成长因子", "不指定类型"
        ])
        form_layout.addRow("因子类型:", self.factor_type_combo)
        
        # 搜索预算
        self.budget_spin = QSpinBox()
        self.budget_spin.setRange(1, 100)
        self.budget_spin.setValue(INITIAL_SEARCH_BUDGET)
        form_layout.addRow("搜索预算:", self.budget_spin)
        
        # 准入阈值
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1, 10)
        self.threshold_spin.setValue(int(EFFECTIVENESS_THRESHOLD))
        form_layout.addRow("准入阈值:", self.threshold_spin)
        
        layout.addLayout(form_layout)
        
        # 按钮
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ 开始搜索")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover { background-color: #2ecc71; }
        """)
        self.start_btn.clicked.connect(self.on_start)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹ 停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover { background-color: #c0392b; }
        """)
        self.stop_btn.clicked.connect(self.on_stop)
        button_layout.addWidget(self.stop_btn)
        
        layout.addLayout(button_layout)
        
        # 其他按钮
        other_btn_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("🗑 清空树")
        self.clear_btn.clicked.connect(self.clear_tree.emit)
        other_btn_layout.addWidget(self.clear_btn)
        
        self.export_btn = QPushButton("💾 导出结果")
        self.export_btn.clicked.connect(self.export_results.emit)
        other_btn_layout.addWidget(self.export_btn)
        
        layout.addLayout(other_btn_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)
    
    def on_start(self):
        """开始搜索"""
        factor_type = self.factor_type_combo.currentText()
        budget = self.budget_spin.value()
        threshold = self.threshold_spin.value()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("搜索中...")
        
        self.start_search.emit(factor_type, budget, threshold)
    
    def on_stop(self):
        """停止搜索"""
        self.stop_search.emit()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("已停止")
    
    def update_progress(self, value: int):
        """更新进度"""
        self.progress_bar.setValue(value)
    
    def reset_state(self):
        """重置状态"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("就绪")
        self.progress_bar.setValue(0)


# =============================================================================
# 主窗口
# =============================================================================

class MCTSMainWindow(QMainWindow):
    """MCTS可视化主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AlphaMCTS 可视化系统")
        self.setGeometry(100, 100, 1600, 900)
        
        self.setup_ui()
        self.setup_connections()
        
        # MCTS相关
        self.mcts: Optional[MCTS] = None
        self.alpha_repo = AlphaLibrary()
        self.current_iteration = 0
        self.total_budget = 0
        self.search_threshold = 2
        
        # 创建后台工作线程
        self.worker = MCTSWorker()
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.setup_worker_connections()
        self.worker_thread.start()
        
        # 重定向stdout
        self.original_stdout = sys.stdout
        sys.stdout = self.StdoutRedirector(self.terminal)
        
        # 存储节点图形项的引用
        self.node_items_map = {}
    
    def setup_worker_connections(self):
        """设置工作线程信号连接"""
        # 工作线程 -> GUI
        self.worker.root_node_ready.connect(self.on_root_node_ready)
        self.worker.iteration_started.connect(self.on_iteration_started)
        self.worker.node_expanded.connect(self.on_node_expanded)
        self.worker.node_evaluated.connect(self.on_node_evaluated)
        self.worker.backprop_done.connect(self.on_backprop_done)
        self.worker.search_finished.connect(self.on_search_finished)
        self.worker.search_error.connect(self.on_search_error)
        self.worker.log_message.connect(self.on_worker_log)
    
    def on_worker_log(self, text: str, log_type: str):
        """处理工作线程日志"""
        self.terminal.append_log(text, log_type)
    
    def on_root_node_ready(self, root_node: AlphaNode):
        """根节点就绪"""
        self.mcts = self.worker.mcts
        self.alpha_repo = self.worker.alpha_repo
        
        # 添加到树形图
        root_item = self.tree_scene.add_node(root_node)
        self.node_items_map[id(root_node)] = root_item
        
        # 更新详情面板
        self.node_detail.update_node(root_node)
        
        # 调整视图
        self.graphics_view.centerOn(root_item)
        self.graphics_view.viewport().update()
    
    def on_iteration_started(self, current: int, total: int):
        """迭代开始"""
        self.current_iteration = current
        progress = int((current / total) * 100)
        self.control_panel.update_progress(progress)
    
    def on_node_expanded(self, parent_node: AlphaNode, new_node: AlphaNode):
        """节点扩展完成"""
        # 找到父节点图形项
        parent_item = self.node_items_map.get(id(parent_node))
        
        # 添加新节点到树
        new_item = self.tree_scene.add_node(new_node, parent_item)
        self.node_items_map[id(new_node)] = new_item
        
        # 更新视图
        self.graphics_view.viewport().update()
        
        # 更新详情面板
        self.node_detail.update_node(new_node)
    
    def on_node_evaluated(self, node: AlphaNode):
        """节点评估完成"""
        # 更新节点显示（评估后可能有新的Q值）
        if id(node) in self.node_items_map:
            self.node_items_map[id(node)].update_color()
        self.graphics_view.viewport().update()
    
    def on_backprop_done(self, node: AlphaNode):
        """反向传播完成"""
        # 更新节点颜色（Q值可能已改变）
        if id(node) in self.node_items_map:
            self.node_items_map[id(node)].update_color()
        self.graphics_view.viewport().update()
    
    def on_search_finished(self):
        """搜索完成"""
        self.terminal.append_log("搜索完成!", "success")
        self.control_panel.reset_state()
        self.status_bar.showMessage(f"搜索完成 | 共发现 {len(self.alpha_repo)} 个优质因子")
    
    def on_search_error(self, error_msg: str):
        """搜索错误"""
        self.terminal.append_log(error_msg, "error")
        QMessageBox.critical(self, "搜索错误", error_msg)
        self.control_panel.reset_state()
    
    class StdoutRedirector:
        """标准输出重定向器"""
        def __init__(self, terminal: TerminalWidget):
            self.terminal = terminal
        
        def write(self, text):
            if text.strip():
                self.terminal.append_log(text.strip())
        
        def flush(self):
            pass
    
    def setup_ui(self):
        """设置UI"""
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 左侧：搜索树视图
        left_splitter = QSplitter(Qt.Horizontal)
        
        # 树形图
        tree_group = QGroupBox("MCTS搜索树")
        tree_layout = QVBoxLayout(tree_group)
        
        self.graphics_view = QGraphicsView()
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.graphics_view.setStyleSheet("background-color: #f5f6fa;")
        
        self.tree_scene = MCTSTreeScene()
        self.graphics_view.setScene(self.tree_scene)
        
        tree_layout.addWidget(self.graphics_view)
        
        # 节点详情面板
        self.node_detail = NodeDetailPanel()
        self.node_detail.setMaximumWidth(350)
        
        left_splitter.addWidget(tree_group)
        left_splitter.addWidget(self.node_detail)
        left_splitter.setSizes([800, 350])
        
        main_layout.addWidget(left_splitter, 2)
        
        # 右侧：控制和终端
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(10)
        
        # 控制面板
        self.control_panel = ControlPanel()
        right_layout.addWidget(self.control_panel)
        
        # 终端
        terminal_group = QGroupBox("运行日志")
        terminal_layout = QVBoxLayout(terminal_group)
        
        self.terminal = TerminalWidget()
        terminal_layout.addWidget(self.terminal)
        
        right_layout.addWidget(terminal_group, 1)
        
        main_layout.addWidget(right_widget, 1)
        
        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪")
    
    def setup_connections(self):
        """设置信号连接"""
        self.control_panel.start_search.connect(self.start_mcts_search)
        self.control_panel.stop_search.connect(self.stop_mcts_search)
        self.control_panel.clear_tree.connect(self.clear_tree)
        self.control_panel.export_results.connect(self.export_results)
        
        self.tree_scene.node_clicked.connect(self.on_node_clicked)
    
    def start_mcts_search(self, factor_type: str, budget: int, threshold: int):
        """开始MCTS搜索"""
        self.terminal.append_log(f"开始搜索: 类型={factor_type}, 预算={budget}, 阈值={threshold}", "info")
        
        # 清空之前的树
        self.clear_tree()
        self.node_items_map.clear()
        
        # 保存参数
        self.total_budget = budget
        self.search_threshold = threshold
        
        # 禁用按钮，显示状态
        self.control_panel.start_btn.setEnabled(False)
        self.control_panel.stop_btn.setEnabled(True)
        self.status_bar.showMessage("初始化中...")
        
        # 临时连接根节点就绪信号（只执行一次）
        def on_root_ready_once(node):
            self.worker.root_node_ready.disconnect(on_root_ready_once)
            QTimer.singleShot(100, lambda: self.start_search_loop(budget, threshold))
        
        self.worker.root_node_ready.connect(on_root_ready_once)
        
        # 在工作线程中初始化根节点
        QTimer.singleShot(100, lambda: self.worker.initialize_root(factor_type))
    
    def start_search_loop(self, budget: int, threshold: int):
        """开始搜索循环"""
        self.status_bar.showMessage(f"搜索中... (0/{budget})")
        # 在工作线程中运行搜索
        QTimer.singleShot(100, lambda: self.worker.run_search(budget, threshold))
    
    def stop_mcts_search(self):
        """停止搜索"""
        self.worker.stop()
        self.terminal.append_log("正在停止...", "warning")
    
    def closeEvent(self, event):
        """关闭事件"""
        # 停止工作线程
        self.worker.stop()
        self.worker_thread.quit()
        self.worker_thread.wait(2000)
        # 恢复stdout
        sys.stdout = self.original_stdout
        event.accept()
    
    def clear_tree(self):
        """清空树"""
        self.tree_scene.clear_tree()
        self.graphics_view.viewport().update()
        self.terminal.append_log("树已清空", "info")
    
    def export_results(self):
        """导出结果"""
        if not self.alpha_repo.alphas:
            QMessageBox.warning(self, "警告", "没有可导出的结果")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "results/mcts_results.json", "JSON文件 (*.json)"
        )
        
        if filename:
            export_elite_factors(self.alpha_repo.alphas, mode="gui_export", save_dir=os.path.dirname(filename))
            self.terminal.append_log(f"结果已导出到: {filename}", "success")
    
    def on_node_clicked(self, node: AlphaNode):
        """节点点击事件"""
        self.node_detail.update_node(node)
    
    def closeEvent(self, event):
        """关闭事件"""
        # 恢复stdout
        sys.stdout = self.original_stdout
        event.accept()


# =============================================================================
# 入口
# =============================================================================

def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 设置全局字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    
    # 创建并显示主窗口
    window = MCTSMainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
