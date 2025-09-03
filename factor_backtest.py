import os
import re
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from tradelearn.query.query import Query
from tradelearn.strategy.examine.examine import Examine
from tradelearn.strategy.backtest import Backtest
from tradelearn.strategy.evaluate.evaluate import Evaluate

from tools.outlier import Outlier
from tools.miss import Miss
from tools.neutralize import Neutralize
from tools.scale import Scale
from tools.label import Label

from strategy import RandomForestStrategy


def returns(df):
    """
    日收益率
    """
    return df.rolling(2).apply(lambda x: x.iloc[-1] / x.iloc[0]) - 1


def ts_sum(df, window=10):
    """
    Wrapper function to estimate rolling sum.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    
    return df.rolling(window).sum()

def sma(df, window=10):
    """
    Wrapper function to estimate SMA.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).mean()

def stddev(df, window=10):
    """
    Wrapper function to estimate rolling standard deviation.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).std()

def correlation(x, y, window=10):
    """
    Wrapper function to estimate rolling corelations.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).corr(y).fillna(0).replace([np.inf, -np.inf], 0)

def covariance(x, y, window=10):
    """
    Wrapper function to estimate rolling covariance.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).cov(y)

def rolling_rank(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The rank of the last value in the array.
    """
    return rankdata(na,method='min')[-1]

def ts_rank(df, window=10):
    """
    Wrapper function to estimate rolling rank.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series rank over the past window days.
    """
    return df.rolling(window).apply(rolling_rank)

def rolling_prod(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The product of the values in the array.
    """
    return np.prod(na)

def product(df, window=10):
    """
    Wrapper function to estimate rolling product.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series product over the past 'window' days.
    """
    return df.rolling(window).apply(rolling_prod)

def ts_min(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).min()

def ts_max(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series max over the past 'window' days.
    """
    return df.rolling(window).max()

def delta(df, period=1):
    """
    Wrapper function to estimate difference.
    :param df: a pandas DataFrame.
    :param period: the difference grade.
    :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
    """
    return df.diff(period)

def delay(df, period=1):
    """
    Wrapper function to estimate lag.
    :param df: a pandas DataFrame.
    :param period: the lag grade.
    :return: a pandas DataFrame with lagged time series
    """
    return df.shift(period)

def rank(df):
    """
    Cross sectional rank
    :param df: a pandas DataFrame.
    :return: a pandas DataFrame with rank along columns.
    """
    return df.rank(axis=1, method='min', pct=True)
    # return df.rank(pct=True)

def scale(df, k=1):
    """
    Scaling time serie.
    :param df: a pandas DataFrame.
    :param k: scaling factor.
    :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
    """
    return df.mul(k).div(np.abs(df).sum())

def ts_argmax(df, window=10):
    """
    Wrapper function to estimate which day ts_max(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmax) + 1 

def ts_argmin(df, window=10):
    """
    Wrapper function to estimate which day ts_min(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmin) + 1

def decay_linear(df, period=10):
    """
    Linear weighted moving average implementation.
    :param df: a pandas DataFrame.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    weights = np.array(range(1, period+1))
    sum_weights = np.sum(weights)
    return df.rolling(period).apply(lambda x: np.sum(weights*x) / sum_weights)

def max(sr1,sr2):
    return np.maximum(sr1, sr2)

def min(sr1,sr2):
    return np.minimum(sr1, sr2)

Operation_Set = {
    "returns(x)": "x的日收益率",
    "ts_sum(x, window)": "x在过去window天的求和值",
    "sma(x, window)": "x在过去window天的均值",
    "stddev(x, window)": "x在过去window天的标准差",
    "correlation(x, y, window)": "x与y在过去window天的相关系数",
    "covariance(x, y, window)": "x与y在过去window天的协方差",
    "ts_rank(x, window)": "x在过去window天的排序",
    "product(x, window)": "x在过去window天的乘积",
    "ts_min(x, window)": "x在过去window天的最小值",
    "ts_max(x, window)": "x在过去window天的最大值",
    "delta(x, period)": "x的period阶差分",
    "delay(x, period)": "x滞后period天",
    "rank(x)": "x的横截面排序（百分比）",
    "scale(x, k)": "x归一化后sum(abs(x))=k",
    "ts_argmax(x, window)": "x在过去window天最大值出现的位置",
    "ts_argmin(x, window)": "x在过去window天最小值出现的位置",
    "decay_linear(x, period)": "x在过去period天的线性加权移动平均",
    "max(x, y)": "x和y的逐元素最大值",
    "min(x, y)": "x和y的逐元素最小值",
}


# 你可以在这里继续扩展别的函数
DEFAULT_FUNCS = {
    # 'Pct': Pct,
    # 'Std': Std,
    # 'Sum': Sum,
    'returns': returns,
    'ts_sum': ts_sum,
    'sma': sma,
    'stddev': stddev,
    'correlation': correlation,
    'covariance': covariance,
    'rolling_rank': rolling_rank,
    'ts_rank': ts_rank,
    'product': product,
    'ts_min': ts_min,
    'ts_max': ts_max,
    'delta': delta,
    'delay': delay,
    'rank': rank,
    'scale': scale,
    'ts_argmax': ts_argmax,
    'ts_argmin': ts_argmin,
    'decay_linear': decay_linear,
    'max': max,
    'min': min,
}


# ---- 解析与求值器 ----
class FormulaParser:
    TOKEN_SPEC = [
        ('NUMBER', r'\d+(\.\d+)?'),
        ('IDENT',  r'[A-Za-z_]\w*'),
        ('RELOP',  r'<=|>=|==|!=|<|>'),   # 关系运算符（注意顺序：长在前）
        ('OP',     r'[+\-*/]'),
        ('QMARK',  r'\?'),
        ('COLON',  r':'),
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
        ('COMMA',  r','),
        ('SKIP',   r'\s+'),
    ]
    MASTER_RE = re.compile('|'.join(f'(?P<{name}>{pat})' for name, pat in TOKEN_SPEC))

    def __init__(self, funcs=None, schema_vars=None):
        self.funcs = funcs or {}
        self.schema_vars = schema_vars or {}

    def eval(self, text: str):
        text = text.replace('·', '*')
        self.tokens = list(self._tokenize(text))
        self.pos = 0
        value = self._expr()
        self._expect_end()
        return value

    # -------------------- Lexer --------------------
    def _tokenize(self, text):
        for m in self.MASTER_RE.finditer(text):
            kind = m.lastgroup
            if kind == 'SKIP':
                continue
            value = m.group()
            if kind == 'NUMBER':
                num = float(value)
                value = int(num) if num.is_integer() else num
            yield (kind, value)

    # -------------------- Cursor helpers --------------------
    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

    def _advance(self):
        tok = self._peek()
        self.pos += 1
        return tok

    def _match(self, *kinds):
        if self._peek()[0] in kinds:
            return self._advance()
        return None

    def _expect(self, kind, value=None):
        tok = self._advance()
        if tok[0] != kind or (value is not None and tok[1] != value):
            raise SyntaxError(f'Expect {kind}{("=" + str(value)) if value is not None else ""}, got {tok}')
        return tok

    def _expect_end(self):
        if self.pos != len(self.tokens):
            raise SyntaxError('Unexpected input at end.')

    # -------------------- Grammar --------------------
    # expr -> conditional (三目最低优先级)
    def _expr(self):
        return self._conditional()

    # conditional -> comparison ('?' expr ':' conditional)?
    # 右结合：a?b:c?d:e 解析为 a?b:(c?d:e)
    def _conditional(self):
        cond = self._comparison()
        tok = self._peek()
        if tok[0] == 'QMARK':
            self._advance()  # '?'
            then_val = self._expr()
            self._expect('COLON')
            else_val = self._conditional()
            return self._ternary(cond, then_val, else_val)
        return cond

    # comparison -> arith (RELOP arith)*
    def _comparison(self):
        left = self._arith()
        while True:
            tok = self._peek()
            if tok[0] == 'RELOP':
                op = self._advance()[1]
                right = self._arith()
                left = self._apply_relop(op, left, right)
            else:
                break
        return left

    # arith -> term (('+'|'-') term)*
    def _arith(self):
        value = self._term()
        while True:
            tok = self._peek()
            if tok[0] == 'OP' and tok[1] in ('+', '-'):
                op = self._advance()[1]
                rhs = self._term()
                value = self._apply_op(op, value, rhs)
            else:
                break
        return value

    # term -> factor (('*'|'/') factor)*
    def _term(self):
        value = self._factor()
        while True:
            tok = self._peek()
            if tok[0] == 'OP' and tok[1] in ('*', '/'):
                op = self._advance()[1]
                rhs = self._factor()
                value = self._apply_op(op, value, rhs)
            else:
                break
        return value

    # factor -> ('+'|'-') factor | '(' expr ')' | NUMBER | IDENT | IDENT '(' args? ')'
    def _factor(self):
        tok = self._peek()

        # 一元 ±
        if tok[0] == 'OP' and tok[1] in ('+', '-'):
            op = self._advance()[1]
            val = self._factor()
            return val if op == '+' else self._neg(val)

        # 括号
        if tok[0] == 'LPAREN':
            self._advance()
            val = self._expr()
            self._expect('RPAREN')
            return val

        # 数字
        if tok[0] == 'NUMBER':
            return self._advance()[1]

        # 标识符：变量 或 函数调用
        if tok[0] == 'IDENT':
            name = self._advance()[1]
            if name in self.schema_vars and (self._peek()[0] != 'LPAREN'):
                return self.schema_vars.get(name, 0)
            elif self._match('LPAREN'):
                args = []
                if self._peek()[0] != 'RPAREN':
                    args.append(self._expr())
                    while self._match('COMMA'):
                        args.append(self._expr())
                self._expect('RPAREN')
                return self._call_func(name, args)

        raise SyntaxError(f'Unexpected token: {tok}')

    # -------------------- Semantics --------------------
    def _apply_op(self, op, lhs, rhs):
        if op == '+':  return lhs + rhs
        if op == '-':  return lhs - rhs
        if op == '*':  return lhs * rhs
        if op == '/':
            # 你原有的除零保护逻辑
            if isinstance(rhs, (pd.Series, pd.DataFrame)):
                rhs = rhs.replace(0, np.nan)
            elif rhs == 0:
                rhs = np.nan
            return lhs / rhs
        raise ValueError(f'Unknown op {op}')

    def _apply_relop(self, op, lhs, rhs):
        if op == '<':  return lhs < rhs
        if op == '<=': return lhs <= rhs
        if op == '>':  return lhs > rhs
        if op == '>=': return lhs >= rhs
        if op == '==': return lhs == rhs
        if op == '!=': return lhs != rhs
        raise ValueError(f'Unknown relop {op}')

    def _neg(self, val):
        return -val

    def _ternary(self, cond, a, b):
        # pandas Series
        if isinstance(cond, pd.Series):
            # 优先使用 a.where 以保持类型与索引
            if hasattr(getattr(a, '__class__', object), 'where') or hasattr(a, 'where'):
                return a.where(cond, b)
            return pd.Series(np.where(cond, a, b), index=cond.index)

        # pandas DataFrame
        if isinstance(cond, pd.DataFrame):
            if hasattr(getattr(a, '__class__', object), 'where') or hasattr(a, 'where'):
                return a.where(cond, b)
            return pd.DataFrame(np.where(cond, a, b), index=cond.index, columns=cond.columns)

        # numpy ndarray
        if isinstance(cond, np.ndarray):
            return np.where(cond, a, b)

        # 标量布尔
        return a if bool(cond) else b

    def _call_func(self, name, args):
        if name not in self.funcs:
            raise NameError(f'Unknown function: {name}')
        return self.funcs[name](*args)


# 自动执行工具函数
def df_factor_process(raw_data):
    print('start processing!')
    post_data = None
    for symbol in raw_data['code'].unique():  # 标的层处理
        data = raw_data.query(f"code == '{symbol}'")
        data = data.set_index('date')

        data['return'] = data['close'].pct_change(periods=5).shift(-1)

        data['date'] = data.index

        data = data.iloc[:-1]
        post_data = pd.concat([post_data, data], axis=0)
    post_data = post_data.reset_index(drop=True)

    fina_data = None
    for date in post_data['date'].unique():  # 时间层处理
        print(f'date being processed: {date}')

        data = post_data.query(f"date == '{date}'")
        t_data = data.set_index('code')
        other_col = ['date', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'adjustflag', 'turn', 'tradestatus',
                        'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ', 'isST', 'cir_a_num', 'vwap',
                        'return', 'ind_code', 'cir_a']  # 只留下 alpha101 因子
        temp = t_data.drop(other_col, axis=1)

        temp = Outlier.winorize_med(temp, scale=5, axis=0)
        temp = Miss.replace_nan_indu(temp, t_data['ind_code'])
        temp = Neutralize.neutralize(temp, t_data[['ind_code']], t_data['cir_a'])
        temp = Scale.standardlize(temp)
        temp = Label.label_by_percent(temp, t_data['return'], pos_prercent=0.3, neg_percent=0.3)  # 可能造成部分股票在时间上不连续

        temp[other_col] = data[other_col].values
        temp['code'] = temp.index
        fina_data = pd.concat([fina_data, temp], axis=0)
    fina_data = fina_data.reset_index(drop=True)
    fina_data = fina_data[~np.isnan(fina_data['label'])]

    fina_data = fina_data.fillna(0)  # 这里已经对缺失值进行了处理，不需要再进行处理

    return fina_data


class FactorBacktest:

    def __init__(self) -> None:
        begin_date = '2017-01-01'
        end_date = '2023-06-26'
        self.raw_data = Query.read_csv(data_path='./data/000300SH.csv', begin=begin_date, end=end_date).iloc[:, 1:]
        stock_data = self.raw_data.pivot(index='date', columns='code')
        self.local_env = locals().copy()
        self.schema_vars = {
            "open": stock_data['open'], # 开盘价
            "high": stock_data['high'], # 最高价
            "low": stock_data['low'], # 最低价
            "close": stock_data['close'], # 收盘价
            "volume": stock_data['volume'], # 成交量
            "returns": returns(stock_data['close']), # 日收益
            "vwap": stock_data['vwap'],  # 成交均价
        }
        self.funcs = DEFAULT_FUNCS
        self.local_env.update(self.schema_vars)
        self.local_env.update(self.funcs)
        self.dummy_parser = FormulaParser(funcs=self.funcs, schema_vars=self.schema_vars)

    def run_backtest_by_eval(self, formula, value_name):
        s_df = eval(formula, globals(), self.local_env)
        res = pd.DataFrame({'date':[], 'code':[]})
        s_df['date'] = s_df.index
        s_df = s_df.melt(id_vars='date', value_vars=s_df.columns.drop('date'), value_name=value_name)
        res = pd.merge(res, s_df, how='outer', on=['date', 'code'])
        s_df = res

        self.raw_data = pd.merge(self.raw_data, s_df, how='left', on=['date', 'code'])
        del s_df

        fina_data = df_factor_process(self.raw_data)
        eval_res = Examine.factor_compare(fina_data)
        return eval_res
    
    def run_backtest(self, formula, value_name):
        
        s_df = self.dummy_parser.eval(formula)
        
        # 处理无穷大值
        s_df = s_df.replace([np.inf, -np.inf], np.nan)
        
        # 将 NaN 替换为 0
        s_df = s_df.fillna(0)

        res = pd.DataFrame({'date':[], 'code':[]})
        s_df['date'] = s_df.index
        s_df = s_df.melt(id_vars='date', value_vars=s_df.columns.drop('date'), value_name=value_name)
        res = pd.merge(res, s_df, how='outer', on=['date', 'code'])
        s_df = res

        self.raw_data = pd.merge(self.raw_data, s_df, how='left', on=['date', 'code'])
        del s_df

        post_data = None
        for symbol in self.raw_data['code'].unique():  # 标的层处理
            data = self.raw_data.query(f"code == '{symbol}'")
            data = data.set_index('date')

            data['return'] = data['close'].pct_change(periods=5).shift(-1)

            data['date'] = data.index

            data = data.iloc[:-1]
            post_data = pd.concat([post_data, data], axis=0)
        post_data = post_data.reset_index(drop=True)

        # fina_data = df_factor_process(self.raw_data)
        eval_res = Examine.factor_compare(post_data)
        return eval_res