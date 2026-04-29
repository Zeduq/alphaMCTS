"""
改进的评估器模块 - 使用类封装消除全局状态

主要改进：
1. 使用 Evaluator 类封装所有评估逻辑
2. 支持缓存机制避免重复计算
3. 更好的错误处理和日志记录
4. 支持训练/测试期数据分离
"""

import numpy as np
import pandas as pd
import random
import warnings
import traceback
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# 导入 Alphalens (仅使用清洗功能)
try:
    from alphalens.utils import get_clean_factor_and_forward_returns
    import empyrical as ep
    ALPHALENS_AVAILABLE = True
except ImportError:
    ALPHALENS_AVAILABLE = False
except Exception:
    ALPHALENS_AVAILABLE = False

from utils.data_structures import AlphaFormula, AlphaNode
from agents.critic_agent import CriticAgent
from factor_calculator import FactorCalculator
from alpha_library.library import AlphaLibrary
from config import (
    MAX_EVAL_SCORE_PER_DIM, EVAL_TEMP, PROMPT_DIR, 
    DATA_DIR
)

# ALPHALENS_AVAILABLE 在模块内定义
try:
    from alphalens.utils import get_clean_factor_and_forward_returns
    import empyrical as ep
    ALPHALENS_AVAILABLE = True
except ImportError:
    ALPHALENS_AVAILABLE = False
except Exception:
    ALPHALENS_AVAILABLE = False


@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    rank_ic_mean: float = 0.0
    rank_ic_std: float = 0.0
    icir: float = 0.0
    turnover: float = 1.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    excess_return: float = 0.0
    information_ratio: float = 0.0


class Evaluator:
    """
    Alpha因子评估器
    
    封装所有评估逻辑，消除全局状态，支持缓存机制
    """
    
    def __init__(self, 
                 data_dir: str = DATA_DIR,
                 train_begin: str = '2017-01-01',
                 train_end: str = '2022-06-26',
                 test_begin: str = '2022-06-27',
                 test_end: str = '2023-06-26',
                 holding_period: int = 5):
        """
        初始化评估器
        
        Args:
            data_dir: 数据目录路径
            train_begin: 训练期开始日期
            train_end: 训练期结束日期
            test_begin: 测试期开始日期
            test_end: 测试期结束日期
            holding_period: 默认持仓周期
        """
        self.data_dir = data_dir
        self.train_begin = train_begin
        self.train_end = train_end
        self.test_begin = test_begin
        self.test_end = test_end
        self.holding_period = holding_period
        
        # 缓存
        self._benchmark_ret: Optional[pd.Series] = None
        self._alpha101_data: Optional[pd.DataFrame] = None
        self._price_data: Dict[str, pd.DataFrame] = {}
        self._factor_cache: Dict[str, pd.Series] = {}
        
        # 初始化Critic Agent
        self.critic_agent = CriticAgent(
            prompt_path=os.path.join(PROMPT_DIR, "overfitting_assessment.txt")
        )
        
        # 初始化因子计算器（训练期）
        try:
            self.calculator = FactorCalculator(
                data_path=os.path.join(data_dir, '000300SH.csv'),
                begin_date=train_begin,
                end_date=train_end,
                holding_period=holding_period
            )
            self.prices = self.calculator.prices
        except Exception as e:
            print(f"[Evaluator] 警告: 核心评估器初始化失败: {e}")
            self.calculator = None
            self.prices = pd.DataFrame()
        
        # 加载辅助数据
        self._load_auxiliary_data()
    
    def _load_auxiliary_data(self):
        """加载基准数据和 Alpha101 数据"""
        print("--- [Evaluator] 正在加载辅助评估数据 ---")
        
        # 加载基准指数 (HS300 Index)
        try:
            bench_path = os.path.join(self.data_dir, 'hs300_index.csv')
            if os.path.exists(bench_path):
                bench_df = pd.read_csv(bench_path, parse_dates=['date']).set_index('date')
                self._benchmark_ret = bench_df['close'].pct_change().fillna(0)
                print(f"[OK] 基准数据加载成功: {len(self._benchmark_ret)} 天")
            else:
                print(f"[WARN] 未找到基准文件: {bench_path}")
        except Exception as e:
            print(f"[FAIL] 基准数据加载失败: {e}")
        
        # 加载 Alpha101 因子库
        try:
            alpha101_path = os.path.join(self.data_dir, 'alpha101.csv')
            if os.path.exists(alpha101_path):
                print("[...] 正在加载 Alpha101 数据...")
                alpha101_df = pd.read_csv(alpha101_path, parse_dates=['date'])
                self._alpha101_data = alpha101_df.set_index(['date', 'code']).sort_index()
                print(f"[OK] Alpha101 数据加载成功: {self._alpha101_data.shape[1]} 个因子")
            else:
                print(f"[WARN] 未找到 Alpha101 文件: {alpha101_path}")
        except Exception as e:
            print(f"[FAIL] Alpha101 数据加载失败: {e}")
    
    def clear_cache(self):
        """清除因子计算缓存"""
        self._factor_cache.clear()
        print("[Evaluator] 缓存已清除")
    
    def calculate_factor(self, formula_str: str, use_cache: bool = True) -> Optional[pd.Series]:
        """
        计算因子值（带缓存）
        
        Args:
            formula_str: 因子公式字符串
            use_cache: 是否使用缓存
            
        Returns:
            因子值的Series，计算失败返回None
        """
        cache_key = formula_str
        
        if use_cache and cache_key in self._factor_cache:
            return self._factor_cache[cache_key]
        
        if self.calculator is None:
            return None
        
        try:
            result = self.calculator.calculate_factor(formula_str)
            if use_cache and result is not None:
                self._factor_cache[cache_key] = result
            return result
        except Exception as e:
            print(f"[Evaluator] 因子计算失败: {e}")
            return None
    
    def get_refinement_dimension(self, scores: Dict[str, float]) -> str:
        """根据分数选择需要优化的维度"""
        refinable_dims = {k: v for k, v in scores.items() if k != "Overfitting Risk"}
        if not refinable_dims:
            return random.choice(list(scores.keys()))
        
        improvement_scores = np.array([MAX_EVAL_SCORE_PER_DIM - v for v in refinable_dims.values()])
        exp_scores = np.exp(improvement_scores / (EVAL_TEMP if EVAL_TEMP > 0 else 1.0))
        probabilities = exp_scores / np.sum(exp_scores)
        return np.random.choice(list(refinable_dims.keys()), p=probabilities)
    
    def _get_refinement_history(self, node: AlphaNode) -> str:
        """获取节点的优化历史"""
        history: List[str] = []
        curr = node
        while curr:
            history.append(f"-> {curr.refinement_summary}")
            curr = curr.parent
        return "\n".join(reversed(history))
    
    def calculate_diversity_score(self, new_factor_stacked: pd.Series, 
                                  alpha_repo: AlphaLibrary) -> float:
        """
        计算多样性分数
        
        如果与现有因子库或 Alpha101 高度相关，则得分低
        """
        if new_factor_stacked is None or new_factor_stacked.empty:
            return MAX_EVAL_SCORE_PER_DIM
        
        max_corr = 0.0
        
        # 准备新因子数据
        try:
            if isinstance(new_factor_stacked, pd.Series):
                new_factor_df = new_factor_stacked.unstack()
            else:
                new_factor_df = new_factor_stacked
        except Exception:
            return 0.0
        
        # 1. 检查 Alpha101 相似度
        if self._alpha101_data is not None:
            try:
                common_dates = new_factor_df.index.intersection(
                    self._alpha101_data.index.levels[0]
                )
                if not common_dates.empty:
                    sample_dates = common_dates[-100:]  # 只取最近100天加速
                    alpha101_slice = self._alpha101_data.loc[sample_dates]
                    new_factor_slice = new_factor_df.loc[sample_dates]
                    
                    new_series = new_factor_slice.stack()
                    new_series.name = 'new_factor'
                    
                    merged = alpha101_slice.join(new_series, how='inner')
                    if not merged.empty:
                        corrs = merged.corrwith(merged['new_factor'])
                        corrs = corrs.drop('new_factor', errors='ignore')
                        if not corrs.empty:
                            max_alpha101_corr = corrs.abs().max()
                            max_corr = max(max_corr, max_alpha101_corr)
            except Exception:
                pass
        
        # 2. 检查内部库相似度
        if alpha_repo.alphas:
            recent_alphas = alpha_repo.alphas[-20:]  # 仅对比最近20个
            for existing_alpha_data in recent_alphas:
                formula_obj = existing_alpha_data.get("formula")
                if formula_obj:
                    try:
                        # 使用缓存避免重复计算
                        existing_factor_res = self.calculate_factor(
                            formula_obj.to_expression_string()
                        )
                        if existing_factor_res is None:
                            continue
                        
                        if isinstance(existing_factor_res, pd.Series):
                            existing_factor_df = existing_factor_res.unstack()
                        elif isinstance(existing_factor_res, pd.DataFrame):
                            existing_factor_df = existing_factor_res
                        else:
                            continue
                        
                        common_index = new_factor_df.index.intersection(existing_factor_df.index)
                        if common_index.empty:
                            continue
                        
                        res = new_factor_df.loc[common_index].corrwith(
                            existing_factor_df.loc[common_index], axis=1
                        )
                        corr = res.abs().mean()
                        
                        if not np.isnan(corr) and corr > max_corr:
                            max_corr = corr
                    except Exception:
                        continue
        
        return MAX_EVAL_SCORE_PER_DIM * (1 - max_corr)
    
    def calculate_manual_metrics(self, factor_data: pd.Series, 
                                 prices: pd.DataFrame, 
                                 period: int = 5) -> Optional[Dict[str, float]]:
        """手动计算指标 (Plan B)"""
        try:
            # 格式清洗
            if isinstance(factor_data, pd.DataFrame):
                factor_df = factor_data
            else:
                factor_df = factor_data.unstack()
            
            factor_df = factor_df.replace([np.inf, -np.inf], np.nan).dropna(how='all')
            if factor_df.empty:
                return None
            
            # 计算未来收益率
            fwd_ret = prices.pct_change(period).shift(-period)
            
            # 对齐
            common_idx = factor_df.index.intersection(fwd_ret.index)
            common_cols = factor_df.columns.intersection(fwd_ret.columns)
            if common_idx.empty:
                return None
            
            factor_df = factor_df.loc[common_idx, common_cols]
            fwd_ret = fwd_ret.loc[common_idx, common_cols]
            
            # Rank IC
            ic_series = factor_df.corrwith(fwd_ret, axis=1, method='spearman')
            rank_ic_mean = ic_series.mean()
            rank_ic_std = ic_series.std()
            icir = rank_ic_mean / rank_ic_std if rank_ic_std != 0 else 0
            
            # 换手率（基于因子自相关性）
            autocorr_series = factor_df.corrwith(factor_df.shift(1), axis=1, method='spearman')
            mean_autocorr = autocorr_series.dropna().mean()
            estimated_turnover = 1.0 - max(0, mean_autocorr) if not np.isnan(mean_autocorr) else 1.0
            
            # 多空策略收益 (Top20% - Bottom20%)
            def get_long_short_ret(row_factor, row_ret):
                if row_factor.count() < 5:
                    return 0.0
                try:
                    q_high = row_factor.quantile(0.8)
                    q_low = row_factor.quantile(0.2)
                    ret_long = row_ret[row_factor >= q_high].mean()
                    ret_short = row_ret[row_factor <= q_low].mean()
                    return ret_long - ret_short
                except:
                    return 0.0
            
            strategy_returns = []
            for dt in factor_df.index:
                ret = get_long_short_ret(factor_df.loc[dt], fwd_ret.loc[dt])
                strategy_returns.append(ret)
            
            strategy_ret_series = pd.Series(strategy_returns, index=factor_df.index).fillna(0)
            
            # 年化指标
            periods_per_year = 252 / period
            
            if ALPHALENS_AVAILABLE:
                ann_ret = ep.stats.annual_return(strategy_ret_series, period='daily', 
                                                  annualization=periods_per_year)
                sharpe = ep.stats.sharpe_ratio(strategy_ret_series, period='daily',
                                                annualization=periods_per_year)
                mdd = ep.stats.max_drawdown(strategy_ret_series)
                ann_excess, ir = self._calculate_excess_stats(strategy_ret_series)
            else:
                # 手动计算
                ann_ret = strategy_ret_series.mean() * periods_per_year
                ret_std = strategy_ret_series.std()
                sharpe = (strategy_ret_series.mean() / ret_std) * np.sqrt(periods_per_year) if ret_std > 0 else 0.0
                
                cum_returns = (1 + strategy_ret_series).cumprod()
                running_max = cum_returns.expanding().max()
                drawdown = (cum_returns - running_max) / running_max
                mdd = drawdown.min()
                
                ann_excess = ann_ret
                ir = sharpe
            
            return {
                "rank_ic_mean": rank_ic_mean if not np.isnan(rank_ic_mean) else 0.0,
                "rank_ic_std": rank_ic_std if not np.isnan(rank_ic_std) else 0.0,
                "icir": icir if not np.isnan(icir) else 0.0,
                "turnover": estimated_turnover,
                "annualized_return": ann_ret,
                "sharpe_ratio": sharpe,
                "max_drawdown": mdd,
                "excess_return": ann_excess,
                "information_ratio": ir
            }
        except Exception as e:
            print(f"[Evaluator] 手动计算指标失败: {e}")
            traceback.print_exc()
            return None
    
    def _calculate_excess_stats(self, strategy_ret_series: pd.Series) -> Tuple[float, float]:
        """计算相对于基准的超额收益指标"""
        if strategy_ret_series.empty or self._benchmark_ret is None:
            return 0.0, 0.0
        
        common_idx = strategy_ret_series.index.intersection(self._benchmark_ret.index)
        if common_idx.empty:
            return 0.0, 0.0
        
        strat_aligned = strategy_ret_series.loc[common_idx]
        bench_aligned = self._benchmark_ret.loc[common_idx]
        
        excess_ret = strat_aligned - bench_aligned
        periods_per_year = 252
        ann_excess_ret = excess_ret.mean() * periods_per_year
        
        std_excess = excess_ret.std()
        if std_excess == 0:
            ir = 0.0
        else:
            ir = excess_ret.mean() / std_excess * np.sqrt(periods_per_year)
        
        return ann_excess_ret, ir
    
    def get_alphalens_metrics(self, factor_data: pd.Series) -> Optional[Dict[str, float]]:
        """使用 Alphalens 计算指标"""
        if not ALPHALENS_AVAILABLE:
            return self.calculate_manual_metrics(factor_data, self.prices, self.holding_period)
        
        try:
            periods = (1, 5, 10)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                factor_data_clean = get_clean_factor_and_forward_returns(
                    factor=factor_data,
                    prices=self.prices,
                    periods=periods,
                    quantiles=5,
                    max_loss=0.40
                )
            
            if factor_data_clean.empty:
                return self.calculate_manual_metrics(factor_data, self.prices, self.holding_period)
            
            period_str = f'{self.holding_period}D'
            target_col = None
            for c in factor_data_clean.columns:
                c_name = c if isinstance(c, str) else c[0]
                if c_name == period_str:
                    target_col = c
                    break
            
            if target_col is None:
                return self.calculate_manual_metrics(factor_data, self.prices, self.holding_period)
            
            # 从清洗后的数据计算指标
            metrics = self._calculate_from_clean_data(factor_data_clean, target_col)
            return metrics if metrics else self.calculate_manual_metrics(
                factor_data, self.prices, self.holding_period
            )
        except Exception as e:
            return self.calculate_manual_metrics(factor_data, self.prices, self.holding_period)
    
    def _calculate_from_clean_data(self, factor_data_clean: pd.DataFrame, 
                                   period_str: str) -> Optional[Dict[str, float]]:
        """从 Alphalens 清洗后的数据计算指标"""
        try:
            # IC
            def src_ic(group):
                return group['factor'].corr(group[period_str], method='spearman')
            
            daily_ic = factor_data_clean.groupby(level='date').apply(src_ic)
            rank_ic_mean = daily_ic.mean()
            rank_ic_std = daily_ic.std()
            icir = rank_ic_mean / rank_ic_std if rank_ic_std != 0 else 0
            
            # Turnover
            factor_wide = factor_data_clean['factor'].unstack()
            autocorr = factor_wide.corrwith(factor_wide.shift(1), axis=1, method='spearman').mean()
            estimated_turnover = 1.0 - max(0, autocorr) if not np.isnan(autocorr) else 1.0
            
            # 策略收益
            df_flat = factor_data_clean.reset_index()
            quantile_returns = df_flat.groupby(['date', 'factor_quantile'])[period_str].mean().unstack()
            
            if 5 in quantile_returns.columns and 1 in quantile_returns.columns:
                long_short_ret = quantile_returns[5] - quantile_returns[1]
            else:
                long_short_ret = pd.Series(0, index=quantile_returns.index)
            long_short_ret = long_short_ret.fillna(0)
            
            # 基础指标
            periods_per_year = 252 / 5
            if '1D' in period_str:
                periods_per_year = 252
            
            ann_ret = ep.stats.annual_return(long_short_ret, period='daily', 
                                              annualization=periods_per_year)
            sharpe = ep.stats.sharpe_ratio(long_short_ret, period='daily',
                                            annualization=periods_per_year)
            mdd = ep.stats.max_drawdown(long_short_ret)
            
            ann_excess, ir = self._calculate_excess_stats(long_short_ret)
            
            return {
                "rank_ic_mean": rank_ic_mean,
                "rank_ic_std": rank_ic_std,
                "icir": icir,
                "turnover": estimated_turnover,
                "annualized_return": ann_ret,
                "sharpe_ratio": sharpe,
                "max_drawdown": mdd,
                "excess_return": ann_excess,
                "information_ratio": ir
            }
        except Exception as e:
            print(f"[Evaluator] 从清洗数据计算指标失败: {e}")
            return None
    
    def evaluate(self, formula: AlphaFormula, node: AlphaNode, 
                 alpha_repo: AlphaLibrary) -> Dict[str, float]:
        """
        核心评估函数
        
        Args:
            formula: Alpha公式
            node: Alpha节点
            alpha_repo: Alpha库
            
        Returns:
            五维评分字典
        """
        scores: Dict[str, float] = {}
        formula_str = formula.to_expression_string()
        
        # 计算因子值
        factor_values = self.calculate_factor(formula_str)
        
        default_metrics = {
            "rank_ic_mean": 0.0, "rank_ic_std": 0.0, "icir": 0.0, "turnover": 1.0,
            "annualized_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0,
            "excess_return": 0.0, "information_ratio": 0.0
        }
        
        # 数据格式转换
        if isinstance(factor_values, pd.DataFrame):
            try:
                factor_values = factor_values.stack()
            except:
                factor_values = None
        
        # 检查数据有效性
        is_invalid = (factor_values is None or factor_values.empty or 
                      factor_values.isnull().values.all())
        
        if is_invalid:
            node.financial_metrics = default_metrics
        else:
            backtest_results = self.get_alphalens_metrics(factor_values)
            node.financial_metrics = backtest_results if backtest_results else default_metrics
        
        metrics = node.financial_metrics
        
        # 评分映射
        rank_ic = metrics.get('rank_ic_mean', 0.0)
        scores["Effectiveness"] = min(MAX_EVAL_SCORE_PER_DIM, max(0.0, abs(rank_ic) * 100))
        
        icir = metrics.get('icir', 0.0)
        scores["Stability"] = min(MAX_EVAL_SCORE_PER_DIM, max(0.0, abs(icir) * 10))
        
        turnover = metrics.get('turnover', 1.0)
        scores["Turnover"] = max(0.0, MAX_EVAL_SCORE_PER_DIM * (1 - min(turnover, 1.0)))
        
        # 多样性
        scores["Diversity"] = self.calculate_diversity_score(factor_values, alpha_repo)
        
        # Critic评分
        history_str = self._get_refinement_history(node)
        critic_output = self.critic_agent.execute(formula=formula, history=history_str)
        if critic_output and 'score' in critic_output:
            try:
                scores["Overfitting Risk"] = float(critic_output.get('score', 5.0))
            except:
                scores["Overfitting Risk"] = 5.0
            node.refinement_summary += f" | Critic: {critic_output.get('reason', 'N/A')}"
        else:
            scores["Overfitting Risk"] = 5.0
        
        # 格式化输出
        for k, v in scores.items():
            try:
                scores[k] = round(float(v), 2)
            except (ValueError, TypeError):
                scores[k] = 0.0
        
        return scores


# 向后兼容：创建默认评估器实例
_default_evaluator: Optional[Evaluator] = None


def get_default_evaluator() -> Evaluator:
    """获取默认评估器实例（单例模式）"""
    global _default_evaluator
    if _default_evaluator is None:
        _default_evaluator = Evaluator()
    return _default_evaluator


def simulate_evaluation(formula: AlphaFormula, node: AlphaNode, 
                        alpha_repo: AlphaLibrary) -> Dict[str, float]:
    """
    向后兼容的评估函数
    
    使用默认评估器实例进行评估
    """
    evaluator = get_default_evaluator()
    return evaluator.evaluate(formula, node, alpha_repo)


def get_refinement_dimension(scores: Dict[str, float]) -> str:
    """向后兼容的函数"""
    evaluator = get_default_evaluator()
    return evaluator.get_refinement_dimension(scores)
