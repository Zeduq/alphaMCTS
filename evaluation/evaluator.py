import numpy as np
import pandas as pd
import random
import warnings
import traceback
import os
from typing import Dict, List

# --- 导入 Alphalens (仅使用清洗功能) ---
try:
    from alphalens.utils import get_clean_factor_and_forward_returns
    import empyrical as ep

    ALPHALENS_AVAILABLE = True
except ImportError:
    print("警告: 未检测到 alphalens 或相关依赖，将默认使用手动计算模式 (Plan B)。")
    ALPHALENS_AVAILABLE = False
except Exception as e:
    print(f"警告: 导入 alphalens 时发生异常: {e}，将默认使用手动计算模式 (Plan B)。")
    ALPHALENS_AVAILABLE = False

from utils.data_structures import AlphaFormula, AlphaNode
from agents.critic_agent import CriticAgent
from config import MAX_EVAL_SCORE_PER_DIM, EVAL_TEMP, PROMPT_DIR
from factor_calculator import FactorCalculator
from alpha_library.library import AlphaLibrary

# --- 全局配置与数据路径 ---
# 假设所有数据都在这个目录下
DATA_DIR = 'D:/AAProject/Data'

# 全局缓存
BENCHMARK_RET = None
ALPHA101_DATA = None

# --- 初始化模块 ---
critic_agent = CriticAgent(prompt_path=os.path.join(PROMPT_DIR, "overfitting_assessment.txt"))


def load_auxiliary_data():
    """加载基准数据和 Alpha101 数据"""
    global BENCHMARK_RET, ALPHA101_DATA

    print(f"--- [Evaluator] 正在加载辅助评估数据 (Benchmark & Alpha101) ---")

    # 1. 加载基准指数 (HS300 Index)
    try:
        bench_path = os.path.join(DATA_DIR, 'hs300_index.csv')
        if os.path.exists(bench_path):
            bench_df = pd.read_csv(bench_path, parse_dates=['date']).set_index('date')
            # 计算基准日收益率
            BENCHMARK_RET = bench_df['close'].pct_change().fillna(0)
            print(f"✅ 基准数据加载成功: {len(BENCHMARK_RET)} 天")
        else:
            print(f"⚠️ 未找到基准文件: {bench_path}，将无法计算超额收益。")
    except Exception as e:
        print(f"❌ 基准数据加载失败: {e}")

    # 2. 加载 Alpha101 因子库 (用于查重/多样性惩罚)
    try:
        alpha101_path = os.path.join(DATA_DIR, 'alpha101.csv')
        if os.path.exists(alpha101_path):
            print("⏳ 正在加载 Alpha101 数据 (可能需要几秒钟)...")
            # 只读取必要的列，优化内存？这里直接全读，假设内存足够
            alpha101_df = pd.read_csv(alpha101_path, parse_dates=['date'])
            # 设为 MultiIndex (date, code) 方便对齐计算
            # 假设 alpha101.csv 中的 code 格式 (如 sh.600000) 与行情数据一致
            ALPHA101_DATA = alpha101_df.set_index(['date', 'code']).sort_index()
            print(f"✅ Alpha101 数据加载成功: {ALPHA101_DATA.shape[1]} 个因子")
        else:
            print(f"⚠️ 未找到 Alpha101 文件: {alpha101_path}，多样性检查将仅限于内部对比。")
    except Exception as e:
        print(f"❌ Alpha101 数据加载失败: {e}")


# 执行加载
load_auxiliary_data()

try:
    calculator = FactorCalculator(
        data_path=os.path.join(DATA_DIR, '000300SH.csv'),
        begin_date='2017-01-01',
        end_date='2023-06-26'
    )
    PRICES = calculator.prices
    HOLDING_PERIOD = calculator.holding_period
except Exception as e:
    print(f"❌ 核心评估器初始化失败，无法加载行情数据: {e}")
    PRICES = pd.DataFrame()
    HOLDING_PERIOD = 5


# --- 辅助计算函数 ---

def calculate_excess_stats(strategy_ret_series: pd.Series):
    """计算相对于基准的超额收益指标"""
    if strategy_ret_series.empty or BENCHMARK_RET is None:
        return 0.0, 0.0  # Annualized Excess Return, IR

    # 对齐日期
    common_idx = strategy_ret_series.index.intersection(BENCHMARK_RET.index)
    if common_idx.empty:
        return 0.0, 0.0

    strat_aligned = strategy_ret_series.loc[common_idx]
    bench_aligned = BENCHMARK_RET.loc[common_idx]

    # 计算每日超额收益
    excess_ret = strat_aligned - bench_aligned

    # 年化超额收益
    periods_per_year = 252  # 假设日频
    ann_excess_ret = ep.stats.annual_return(excess_ret, period='daily', annualization=periods_per_year)

    # 信息比率 (IR) = 超额收益均值 / 超额收益波动率
    # 注意：这里不用 ep.stats.information_ratio 因为它需要传入 benchmark series，逻辑一样但我们手算更透明
    std_excess = excess_ret.std()
    if std_excess == 0:
        ir = 0.0
    else:
        ir = excess_ret.mean() / std_excess * np.sqrt(periods_per_year)

    return ann_excess_ret, ir


def get_refinement_dimension(scores: Dict[str, float]) -> str:
    refinable_dims = {k: v for k, v in scores.items() if k != "Overfitting Risk"}
    if not refinable_dims:
        return random.choice(list(scores.keys()))
    improvement_scores = np.array([MAX_EVAL_SCORE_PER_DIM - v for v in refinable_dims.values()])
    exp_scores = np.exp(improvement_scores / (EVAL_TEMP if EVAL_TEMP > 0 else 1.0))
    probabilities = exp_scores / np.sum(exp_scores)
    return np.random.choice(list(refinable_dims.keys()), p=probabilities)


def _get_refinement_history(node: AlphaNode) -> str:
    history: List[str] = []
    curr = node
    while curr:
        history.append(f"-> {curr.refinement_summary}")
        curr = curr.parent
    return "\n".join(reversed(history))


def calculate_diversity_score(new_factor_stacked: pd.Series, alpha_repo: AlphaLibrary) -> float:
    """
    计算多样性分数。
    规则：如果与现有因子库 OR Alpha101 因子库高度相关，则得分低。
    """
    if new_factor_stacked is None or new_factor_stacked.empty:
        return MAX_EVAL_SCORE_PER_DIM

    max_corr = 0.0

    # 准备新因子数据 (Time x Asset)
    try:
        if isinstance(new_factor_stacked, pd.Series):
            new_factor_df = new_factor_stacked.unstack()
        else:
            new_factor_df = new_factor_stacked
    except Exception:
        return 0.0

    # 1. 检查 Alpha101 相似度 (如果可用)
    if ALPHA101_DATA is not None:
        try:
            # 将新因子转为 Series 并命名，以便 join
            # new_factor_stacked index 是 (date, code)
            # ALPHA101_DATA index 也是 (date, code)

            # 为了速度，我们只取公共时间段的数据切片
            common_dates = new_factor_df.index.intersection(ALPHA101_DATA.index.levels[0])
            if not common_dates.empty:
                # 这是一个优化：只取部分时间点计算相关性，避免全量计算太慢
                # 例如只取最近 100 天
                sample_dates = common_dates[-100:]

                # 获取切片数据
                alpha101_slice = ALPHA101_DATA.loc[sample_dates]

                # 构建新因子切片 (需转回 stack 格式以匹配 join 或直接用 dataframe corrwith)
                new_factor_slice = new_factor_df.loc[sample_dates]

                # 注意：ALPHA101_DATA 是 stack 形式的 MultiIndex DataFrame (列是因子名)
                # new_factor_slice 是 wide 形式 (行时间，列资产)
                # 我们需要把 new_factor_slice 变成 wide 形式与 ALPHA101 里的每个列算 corrwith?
                # 不，ALPHA101_DATA 已经是所有因子的集合。
                # 最快的方法：把 alpha101 unstack 成 wide，然后 corrwith。
                # 但 alpha101 有 101 列，unstack 会很慢。

                # 替代方案：循环 Alpha101 的列 (虽然笨但稳健)
                # 或者：只随机抽查 10 个 Alpha101 因子？
                # 这里我们尝试稍微高效点的方法：

                # 将 Alpha101 slice unstack 可能会很大，但只取 100 天应该还好
                # alpha101_slice (rows=100*300=30000, cols=101)
                # unstack 后 (rows=100, cols=300*101) -> 太大了。

                # 正确做法：保持 Long Format 计算相关性
                # 把 new_factor 变成 Long Format Series
                new_series = new_factor_slice.stack()
                new_series.name = 'new_factor'

                # 合并
                merged = alpha101_slice.join(new_series, how='inner')

                if not merged.empty:
                    # 计算 'new_factor' 列与其他所有列的相关性
                    corrs = merged.corrwith(merged['new_factor'])
                    # 排除自己
                    corrs = corrs.drop('new_factor', errors='ignore')
                    if not corrs.empty:
                        max_alpha101_corr = corrs.abs().max()
                        if max_alpha101_corr > max_corr:
                            max_corr = max_alpha101_corr
                            # print(f"  [Diversity] 发现与 Alpha101 高度相关: {max_corr:.2f}")

        except Exception as e:
            # print(f"Alpha101 多样性检查出错: {e}")
            pass

    # 2. 检查内部库相似度 (Existing Logic)
    if alpha_repo.alphas:
        recent_alphas = alpha_repo.alphas[-20:]  # 仅对比最近 20 个，加速
        for existing_alpha_data in recent_alphas:
            formula_obj = existing_alpha_data.get("formula")
            if formula_obj:
                try:
                    # 缓存优化点：实际应缓存计算结果
                    existing_factor_res = calculator.calculate_factor(formula_obj.to_expression_string())
                    if isinstance(existing_factor_res, pd.Series):
                        existing_factor_df = existing_factor_res.unstack()
                    elif isinstance(existing_factor_res, pd.DataFrame):
                        existing_factor_df = existing_factor_res
                    else:
                        continue

                    common_index = new_factor_df.index.intersection(existing_factor_df.index)
                    if common_index.empty: continue

                    res = new_factor_df.loc[common_index].corrwith(existing_factor_df.loc[common_index], axis=1)
                    corr = res.abs().mean()

                    if not np.isnan(corr) and corr > max_corr:
                        max_corr = corr
                except Exception:
                    continue

    return MAX_EVAL_SCORE_PER_DIM * (1 - max_corr)


# --- 指标计算逻辑 ---

def get_manual_metrics(factor_data: pd.Series, prices: pd.DataFrame, period: int = 5):
    """Plan B: 手动计算指标 (含 Benchmark 对比)"""
    try:
        # 1. 格式清洗
        if isinstance(factor_data, pd.DataFrame):
            factor_df = factor_data
        else:
            factor_df = factor_data.unstack()

        factor_df = factor_df.replace([np.inf, -np.inf], np.nan).dropna(how='all')
        if factor_df.empty: return None

        # 2. 计算未来收益率
        fwd_ret = prices.pct_change(period).shift(-period)

        # 3. 对齐
        common_idx = factor_df.index.intersection(fwd_ret.index)
        common_cols = factor_df.columns.intersection(fwd_ret.columns)
        if common_idx.empty: return None

        factor_df = factor_df.loc[common_idx, common_cols]
        fwd_ret = fwd_ret.loc[common_idx, common_cols]

        # 4. Rank IC
        ic_series = factor_df.corrwith(fwd_ret, axis=1, method='spearman')
        rank_ic_mean = ic_series.mean()
        rank_ic_std = ic_series.std()
        icir = rank_ic_mean / rank_ic_std if rank_ic_std != 0 else 0

        # 5. 简易策略收益 (Top - Bottom)
        def get_top_bottom_ret(row_factor, row_ret):
            if row_factor.count() < 5: return 0.0
            try:
                # 简单的分位数分割
                q_high = row_factor.quantile(0.8)
                q_low = row_factor.quantile(0.2)
                ret_long = row_ret[row_factor >= q_high].mean()
                ret_short = row_ret[row_factor <= q_low].mean()
                return ret_long - ret_short
            except:
                return 0.0

        # 逐日计算策略收益 (这比 groupby 慢但无需 concat)
        strategy_daily_ret_list = []
        for dt in factor_df.index:
            ret = get_top_bottom_ret(factor_df.loc[dt], fwd_ret.loc[dt])
            strategy_daily_ret_list.append(ret)

        strategy_ret_series = pd.Series(strategy_daily_ret_list, index=factor_df.index).fillna(0)

        # 6. 计算年化指标 & 超额收益
        periods_per_year = 252 / period
        ann_ret = ep.stats.annual_return(strategy_ret_series, period='daily', annualization=periods_per_year)
        sharpe = ep.stats.sharpe_ratio(strategy_ret_series, period='daily', annualization=periods_per_year)
        mdd = ep.stats.max_drawdown(strategy_ret_series)

        # 新增: 超额收益计算
        ann_excess, ir = calculate_excess_stats(strategy_ret_series)

        return {
            "rank_ic_mean": rank_ic_mean if not np.isnan(rank_ic_mean) else 0.0,
            "rank_ic_std": rank_ic_std if not np.isnan(rank_ic_std) else 0.0,
            "icir": icir if not np.isnan(icir) else 0.0,
            "turnover": 0.5,
            "annualized_return": ann_ret,
            "sharpe_ratio": sharpe,
            "max_drawdown": mdd,
            "excess_return": ann_excess,  # 新增
            "information_ratio": ir  # 新增
        }
    except Exception as e:
        print(f"Plan B 手动计算失败: {e}")
        traceback.print_exc()
        return None


def calculate_metrics_manually_from_clean_data(factor_data_clean: pd.DataFrame, period_str: str):
    """从 Alphalens 清洗后的数据计算指标 (含 Benchmark 对比)"""
    try:
        # 1. IC
        def src_ic(group):
            return group['factor'].corr(group[period_str], method='spearman')

        daily_ic = factor_data_clean.groupby(level='date').apply(src_ic)
        rank_ic_mean = daily_ic.mean()
        rank_ic_std = daily_ic.std()
        icir = rank_ic_mean / rank_ic_std if rank_ic_std != 0 else 0

        # 2. Turnover (估算)
        factor_wide = factor_data_clean['factor'].unstack()
        autocorr = factor_wide.corrwith(factor_wide.shift(1), axis=1, method='spearman').mean()
        estimated_turnover = 1.0 - max(0, autocorr) if not np.isnan(autocorr) else 1.0

        # 3. 策略收益
        df_flat = factor_data_clean.reset_index()
        quantile_returns = df_flat.groupby(['date', 'factor_quantile'])[period_str].mean().unstack()

        if 5 in quantile_returns.columns and 1 in quantile_returns.columns:
            long_short_ret = quantile_returns[5] - quantile_returns[1]
        else:
            long_short_ret = pd.Series(0, index=quantile_returns.index)
        long_short_ret = long_short_ret.fillna(0)

        # 4. 基础指标
        periods_per_year = 252 / 5
        if '1D' in period_str: periods_per_year = 252

        ann_ret = ep.stats.annual_return(long_short_ret, period='daily', annualization=periods_per_year)
        sharpe = ep.stats.sharpe_ratio(long_short_ret, period='daily', annualization=periods_per_year)
        mdd = ep.stats.max_drawdown(long_short_ret)

        # 5. 新增: 超额收益计算
        ann_excess, ir = calculate_excess_stats(long_short_ret)

        return {
            "rank_ic_mean": rank_ic_mean,
            "rank_ic_std": rank_ic_std,
            "icir": icir,
            "turnover": estimated_turnover,
            "annualized_return": ann_ret,
            "sharpe_ratio": sharpe,
            "max_drawdown": mdd,
            "excess_return": ann_excess,  # 新增
            "information_ratio": ir  # 新增
        }

    except Exception as e:
        print(f"手动计算指标失败: {e}")
        return None


def get_alphalens_metrics(factor_data: pd.Series):
    if not ALPHALENS_AVAILABLE:
        return get_manual_metrics(factor_data, PRICES, HOLDING_PERIOD)

    try:
        periods = (1, 5, 10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            factor_data_clean = get_clean_factor_and_forward_returns(
                factor=factor_data,
                prices=PRICES,
                periods=periods,
                quantiles=5,
                max_loss=0.40
            )

        if factor_data_clean.empty:
            return get_manual_metrics(factor_data, PRICES, HOLDING_PERIOD)

        period_str = f'{HOLDING_PERIOD}D'
        target_col = None
        # 寻找收益列名
        for c in factor_data_clean.columns:
            c_name = c if isinstance(c, str) else c[0]
            if c_name == period_str:
                target_col = c
                break

        if target_col is None:
            # Fallback search
            for c in factor_data_clean.columns:
                c_name = c if isinstance(c, str) else c[0]
                if 'D' in c_name and 'factor' not in c_name:
                    target_col = c
                    period_str = c_name
                    break

        if target_col is None:
            return get_manual_metrics(factor_data, PRICES, HOLDING_PERIOD)

        # 使用手动计算逻辑 (含 Benchmark)
        metrics = calculate_metrics_manually_from_clean_data(factor_data_clean, target_col)

        if metrics:
            return metrics
        else:
            return get_manual_metrics(factor_data, PRICES, HOLDING_PERIOD)

    except Exception as e:
        return get_manual_metrics(factor_data, PRICES, HOLDING_PERIOD)


# --- 核心评估函数 ---

def simulate_evaluation(formula: AlphaFormula, node: AlphaNode, alpha_repo: AlphaLibrary) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    formula_str = formula.to_expression_string()

    factor_values_stacked = None
    try:
        factor_values_stacked = calculator.calculate_factor(formula_str)
    except Exception as e:
        print(f"因子计算过程出错: {e}")

    default_metrics = {
        "rank_ic_mean": 0.0, "rank_ic_std": 0.0, "icir": 0.0, "turnover": 1.0,
        "annualized_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0,
        "excess_return": 0.0, "information_ratio": 0.0
    }

    if isinstance(factor_values_stacked, pd.DataFrame):
        try:
            factor_values_stacked = factor_values_stacked.stack()
        except:
            factor_values_stacked = None

    is_invalid = False
    if factor_values_stacked is None or factor_values_stacked.empty:
        is_invalid = True
    elif factor_values_stacked.isnull().values.all():
        is_invalid = True

    if is_invalid:
        node.financial_metrics = default_metrics
    else:
        backtest_results = get_alphalens_metrics(factor_values_stacked)
        if backtest_results:
            node.financial_metrics = backtest_results
        else:
            node.financial_metrics = default_metrics

    metrics = node.financial_metrics

    # 评分映射 (这里可以根据需要修改，比如 Effectiveness 结合 Excess Return)
    rank_ic = metrics.get('rank_ic_mean', 0.0)
    scores["Effectiveness"] = min(MAX_EVAL_SCORE_PER_DIM, max(0.0, abs(rank_ic) * 100))

    icir = metrics.get('icir', 0.0)
    scores["Stability"] = min(MAX_EVAL_SCORE_PER_DIM, max(0.0, abs(icir) * 10))

    turnover = metrics.get('turnover', 1.0)
    scores["Turnover"] = max(0.0, MAX_EVAL_SCORE_PER_DIM * (1 - min(turnover, 1.0)))

    # 多样性计算 (含 Alpha101)
    scores["Diversity"] = calculate_diversity_score(factor_values_stacked, alpha_repo)

    # Critic 评分
    history_str = _get_refinement_history(node)
    critic_output = critic_agent.execute(formula=formula, history=history_str)
    if critic_output and 'score' in critic_output:
        try:
            scores["Overfitting Risk"] = float(critic_output.get('score', 5.0))
        except:
            scores["Overfitting Risk"] = 5.0
        node.refinement_summary += f" | Critic: {critic_output.get('reason', 'N/A')}"
    else:
        scores["Overfitting Risk"] = 5.0

    for k, v in scores.items():
        try:
            scores[k] = round(float(v), 2)
        except (ValueError, TypeError):
            scores[k] = 0.0

    return scores