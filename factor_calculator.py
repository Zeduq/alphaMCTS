import re
import numpy as np
import pandas as pd
from scipy.stats import rankdata


# --- 核心算子函数 ---

def safe_log(df):
    """自定义的安全log函数，处理非正数。"""
    # 替换 0 或 负数 为 NaN，或者截断
    # 这里选择截断到一个极小正数，防止 log 报错
    epsilon = 1e-10
    # 确保是 DataFrame 或 Series
    if hasattr(df, 'clip'):
        return np.log(df.clip(lower=epsilon))
    return np.log(np.maximum(df, epsilon))


def returns(df):
    # 计算收益率 (Time-Series)
    return df.pct_change(1)


def ts_sum(df, window=10):
    return df.rolling(window).sum()


def sma(df, window=10):
    return df.rolling(window).mean()


def stddev(df, window=10):
    return df.rolling(window).std()


def correlation(x, y, window=10):
    # 处理 rolling corr 可能产生的除零或无穷大
    return x.rolling(window).corr(y).fillna(0).replace([np.inf, -np.inf], 0)


def covariance(x, y, window=10):
    return x.rolling(window).cov(y)


def rolling_rank(na):
    # 辅助函数：计算滚动窗口末尾值的排名
    return rankdata(na, method='min')[-1]


def ts_rank(df, window=10):
    return df.rolling(window).apply(rolling_rank, raw=True)


def product(df, window=10):
    # 滚动乘积
    return df.rolling(window).apply(np.prod, raw=True)


def ts_min(df, window=10):
    return df.rolling(window).min()


def ts_max(df, window=10):
    return df.rolling(window).max()


def delta(df, period=1):
    return df.diff(period)


def delay(df, period=1):
    return df.shift(period)


def rank(df):
    # 截面排名 (Cross-Sectional Rank)
    # axis=1 表示对每一行（每一天）的所有列（股票）进行排名
    return df.rank(axis=1, method='min', pct=True)


def scale(df, k=1):
    """
    截面缩放 (Cross-Sectional Scale)
    使每天的因子绝对值之和为 k (默认1)。
    修复了原版中使用 sum(axis=0) 导致的未来函数和 Series 比较报错。
    """
    # axis=1: 按行（每天）求绝对值之和
    daily_abs_sum = np.abs(df).sum(axis=1)

    # 将和为0的（全空行）替换为NaN，避免除以0
    daily_abs_sum = daily_abs_sum.replace(0, np.nan)

    # div(axis=0): 每一行除以该行的 sum
    return df.div(daily_abs_sum, axis=0).mul(k)


def ts_argmax(df, window=10):
    # 返回最大值所在的索引位置 (1-based)
    return df.rolling(window).apply(np.argmax, raw=True) + 1


def ts_argmin(df, window=10):
    # 返回最小值所在的索引位置 (1-based)
    return df.rolling(window).apply(np.argmin, raw=True) + 1


def decay_linear(df, period=10):
    # 线性衰减加权平均
    weights = np.arange(1, period + 1)
    sum_weights = np.sum(weights)

    def weighted_mean(x):
        return np.sum(weights * x) / sum_weights

    return df.rolling(period).apply(weighted_mean, raw=True)


# 增强的除法函数
def safe_divide(x, y):
    # 如果分母是 DataFrame/Series，替换 0 为 NaN
    if hasattr(y, 'replace'):
        return x / y.replace(0, np.nan)
    # 如果分母是标量
    if y == 0:
        return x * np.nan
    return x / y


# 默认函数库
DEFAULT_FUNCS = {
    'ts_mean': sma, 'ts_std': stddev, 'ts_rank': ts_rank, 'ts_corr': correlation,
    'ts_delta': delta, 'rank': rank, 'scale': scale,
    'log': safe_log,
    'abs': np.abs,
    'sign': np.sign,
    'add': lambda x, y: x + y,
    'subtract': lambda x, y: x - y,
    'multiply': lambda x, y: x * y,
    'divide': safe_divide,  # 使用增强版除法
    'ts_sum': ts_sum, 'sma': sma, 'stddev': stddev, 'correlation': correlation,
    'covariance': covariance, 'product': product, 'ts_min': ts_min, 'ts_max': ts_max,
    'delay': delay, 'ts_argmax': ts_argmax, 'ts_argmin': ts_argmin, 'decay_linear': decay_linear,
}


# --- 公式解析器 ---
class FormulaParser:
    TOKEN_SPEC = [
        ('NUMBER', r'\d+(\.\d+)?'),
        ('IDENT', r'[A-Za-z_]\w*'),
        ('OP', r'[+\-*/]'),
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
        ('COMMA', r','),
        ('SKIP', r'\s+'),
    ]
    MASTER_RE = re.compile('|'.join(f'(?P<{name}>{pat})' for name, pat in TOKEN_SPEC))

    def __init__(self, funcs=None, schema_vars=None):
        self.funcs = funcs or {}
        self.schema_vars = schema_vars or {}

    def eval(self, text: str):
        self.tokens = list(self._tokenize(text))
        self.pos = 0
        value = self._expr()
        self._expect_end()
        return value

    def _tokenize(self, text):
        for m in self.MASTER_RE.finditer(text):
            kind = m.lastgroup
            if kind == 'SKIP': continue
            value = m.group()
            if kind == 'NUMBER':
                num = float(value)
                value = int(num) if num.is_integer() else num
            yield (kind, value)

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

    def _advance(self):
        tok = self._peek();
        self.pos += 1;
        return tok

    def _expect(self, kind):
        tok = self._advance()
        if tok[0] != kind: raise SyntaxError(f'Expect {kind}, got {tok}')
        return tok

    def _expect_end(self):
        if self.pos != len(self.tokens): raise SyntaxError('Unexpected input at end.')

    def _expr(self):
        return self._arith()

    def _arith(self):
        value = self._term()
        while True:
            tok = self._peek()
            if tok[0] == 'OP' and tok[1] in ('+', '-'):
                op = self._advance()[1];
                rhs = self._term();
                value = self._apply_op(op, value, rhs)
            else:
                break
        return value

    def _term(self):
        value = self._factor()
        while True:
            tok = self._peek()
            if tok[0] == 'OP' and tok[1] in ('*', '/'):
                op = self._advance()[1];
                rhs = self._factor();
                value = self._apply_op(op, value, rhs)
            else:
                break
        return value

    def _factor(self):
        tok = self._peek()
        if tok[0] == 'OP' and tok[1] in ('+', '-'):
            op = self._advance()[1];
            val = self._factor();
            return val if op == '+' else -val
        if tok[0] == 'LPAREN':
            self._advance();
            val = self._expr();
            self._expect('RPAREN');
            return val
        if tok[0] == 'NUMBER': return self._advance()[1]
        if tok[0] == 'IDENT':
            name = self._advance()[1]
            if self._peek()[0] == 'LPAREN':
                self._advance();
                args = []
                if self._peek()[0] != 'RPAREN':
                    args.append(self._expr())
                    while self._peek()[0] == 'COMMA': self._advance(); args.append(self._expr())
                self._expect('RPAREN');
                return self._call_func(name, args)
            return self.schema_vars.get(name)
        raise SyntaxError(f'Unexpected token: {tok}')

    def _apply_op(self, op, lhs, rhs):
        if op == '+': return lhs + rhs;
        if op == '-': return lhs - rhs
        if op == '*': return lhs * rhs;
        if op == '/':
            # 使用安全的除法逻辑
            return safe_divide(lhs, rhs)
        raise ValueError(f'Unknown op {op}')

    def _call_func(self, name, args):
        if name in self.funcs:
            return self.funcs[name](*args)
        raise NameError(f'Unknown function: {name}')


# --- 因子计算器封装类 ---
class FactorCalculator:
    def __init__(self, data_path: str = 'D:/AAProject/Data/000300SH.csv',
                 begin_date: str = '2017-01-01', end_date: str = '2023-06-26'):
        print("--- [FactorCalculator] 初始化因子计算器 ---")
        print(f"--- [FactorCalculator] 正在从 {data_path} 加载数据... ---")
        try:
            # 读取数据
            raw_data = pd.read_csv(data_path, parse_dates=['date'], dtype={'code': str}).iloc[:, 1:]
            raw_data = raw_data[(raw_data['date'] >= begin_date) & (raw_data['date'] <= end_date)]
            stock_data = raw_data.pivot(index='date', columns='code')

            # 确保MultiIndex的level 1是'code'
            stock_data.columns = stock_data.columns.set_levels(stock_data.columns.levels[1], level=1)

            self.schema_vars = {
                "open": stock_data['open'], "high": stock_data['high'], "low": stock_data['low'],
                "close": stock_data['close'], "volume": stock_data['volume'], "vwap": stock_data['vwap'],
            }

            # 准备 Alphalens 需要的 assets 和 prices 数据
            self.assets = stock_data['close'].columns
            self.prices = stock_data['close']

            self.holding_period = 5
            self.parser = FormulaParser(funcs=DEFAULT_FUNCS, schema_vars=self.schema_vars)
            print("--- [FactorCalculator] 因子计算器初始化完成 ---")

        except Exception as e:
            print(f"--- [FactorCalculator] 错误: 数据加载失败: {e} ---")
            raise

    def calculate_factor(self, formula_str: str) -> pd.DataFrame:
        try:
            factor_values = self.parser.eval(formula_str)

            # 如果返回的是标量（例如所有股票都一样），尝试广播（虽然不太可能）
            if isinstance(factor_values, (int, float)):
                print(f"警告: 因子计算结果为标量 {factor_values}，非DataFrame")
                return None

            factor_values = factor_values.replace([np.inf, -np.inf], np.nan)

            # 关键：确保因子值的索引和列与价格数据一致
            factor_values = factor_values.reindex(index=self.prices.index, columns=self.prices.columns)

            # Alphalens 偏好 (date, asset) 格式的 MultiIndex
            stacked_factor = factor_values.stack()
            stacked_factor.index.names = ['date', 'asset']
            return stacked_factor

        except Exception as e:
            print(f"错误: 因子计算失败 for formula: {formula_str}\nError details: {e}")
            return None