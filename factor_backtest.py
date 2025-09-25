# alphaMCTS/factor_backtest.py

import re
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, ConstantInputWarning


# --- 因子计算函数 ---
def returns(df):
    return df.rolling(2).apply(lambda x: x.iloc[-1] / x.iloc[0]) - 1

def ts_sum(df, window=10):
    return df.rolling(window).sum()

def sma(df, window=10):
    return df.rolling(window).mean()

def stddev(df, window=10):
    return df.rolling(window).std()

def correlation(x, y, window=10):
    return x.rolling(window).corr(y).fillna(0).replace([np.inf, -np.inf], 0)

def covariance(x, y, window=10):
    return x.rolling(window).cov(y)

def rolling_rank(na):
    return rankdata(na, method='min')[-1]

def ts_rank(df, window=10):
    return df.rolling(window).apply(rolling_rank)

def product(df, window=10):
    return df.rolling(window).apply(np.prod)

def ts_min(df, window=10):
    return df.rolling(window).min()

def ts_max(df, window=10):
    return df.rolling(window).max()

def delta(df, period=1):
    return df.diff(period)

def delay(df, period=1):
    return df.shift(period)

def rank(df):
    return df.rank(axis=1, method='min', pct=True)

def scale(df, k=1):
    return df.mul(k).div(np.abs(df).sum())

def ts_argmax(df, window=10):
    return df.rolling(window).apply(np.argmax) + 1

def ts_argmin(df, window=10):
    return df.rolling(window).apply(np.argmin) + 1

def decay_linear(df, period=10):
    weights = np.array(range(1, period + 1))
    sum_weights = np.sum(weights)
    return df.rolling(period).apply(lambda x: np.sum(weights * x) / sum_weights)


DEFAULT_FUNCS = {
    'ts_mean': sma, 'ts_std': stddev, 'ts_rank': ts_rank, 'ts_corr': correlation,
    'ts_delta': delta, 'rank': rank, 'scale': scale, 'log': np.log, 'abs': np.abs,
    'sign': np.sign, 'add': lambda x, y: x + y, 'subtract': lambda x, y: x - y,
    'multiply': lambda x, y: x * y, 'divide': lambda x, y: x / y.replace(0, np.nan),
    'ts_sum': ts_sum, 'sma': sma, 'stddev': stddev, 'correlation': correlation,
    'covariance': covariance, 'product': product, 'ts_min': ts_min, 'ts_max': ts_max,
    'delay': delay, 'ts_argmax': ts_argmax, 'ts_argmin': ts_argmin, 'decay_linear': decay_linear,
}


# ---- 递归下降解析器 ----
class FormulaParser:
    TOKEN_SPEC = [
        ('NUMBER', r'\d+(\.\d+)?'), ('IDENT', r'[A-Za-z_]\w*'), ('OP', r'[+\-*/]'),
        ('LPAREN', r'\('), ('RPAREN', r'\)'), ('COMMA', r','), ('SKIP', r'\s+'),
    ]
    MASTER_RE = re.compile('|'.join(f'(?P<{name}>{pat})' for name, pat in TOKEN_SPEC))

    def __init__(self, funcs=None, schema_vars=None):
        self.funcs = funcs or {};
        self.schema_vars = schema_vars or {}

    def eval(self, text: str):
        self.tokens = list(self._tokenize(text));
        self.pos = 0
        value = self._expr();
        self._expect_end();
        return value

    def _tokenize(self, text):
        for m in self.MASTER_RE.finditer(text):
            kind = m.lastgroup
            if kind == 'SKIP': continue
            value = m.group()
            if kind == 'NUMBER':
                num = float(value);
                value = int(num) if num.is_integer() else num
            yield (kind, value)

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

    def _advance(self):
        tok = self._peek(); self.pos += 1; return tok

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
        if op == '/': return lhs / rhs.replace(0, np.nan)
        raise ValueError(f'Unknown op {op}')

    def _call_func(self, name, args):
        if name not in self.funcs:
            if hasattr(pd.DataFrame, name):
                df = args[0];
                method = getattr(df, name);
                return method(*args[1:])
            raise NameError(f'Unknown function: {name}')
        return self.funcs[name](*args)


# ---- 回测引擎核心 ----
class FactorBacktest:  # Changed from BacktestEngine to FactorBacktest
    def __init__(self, data_path: str = './data/000300SH.csv', begin_date: str = '2017-01-01',
                 end_date: str = '2023-06-26'):
        print("--- 初始化回测引擎 ---")
        print(f"--- 正在从 {data_path} 加载数据... ---")
        # Ensure 'date' column is parsed as datetime and 'code' as string
        raw_data = pd.read_csv(data_path, parse_dates=['date'], dtype={'code': str}).iloc[:, 1:]
        raw_data = raw_data[(raw_data['date'] >= begin_date) & (raw_data['date'] <= end_date)]
        stock_data = raw_data.pivot(index='date', columns='code')
        self.schema_vars = {
            "open": stock_data['open'], "high": stock_data['high'], "low": stock_data['low'],
            "close": stock_data['close'], "volume": stock_data['volume'], "vwap": stock_data['vwap'],
        }
        # Calculate future returns for IC calculation (5-day forward return)
        self.future_returns = stock_data['close'].pct_change(periods=5).shift(-5)
        self.parser = FormulaParser(funcs=DEFAULT_FUNCS, schema_vars=self.schema_vars)
        print("--- 回测引擎初始化完成 ---")

    def calculate_factor(self, formula_str: str) -> pd.DataFrame:
        try:
            factor_values = self.parser.eval(formula_str)
            factor_values = factor_values.replace([np.inf, -np.inf], np.nan)
            return factor_values.fillna(0)
        except Exception as e:
            print(f"错误: 因子计算失败 for formula: {formula_str}\nError details: {e}")
            return None

    @staticmethod
    def calculate_rank_ic(factor_values: pd.DataFrame, future_returns: pd.DataFrame):
        common_index = factor_values.index.intersection(future_returns.index)
        factor_values = factor_values.loc[common_index];
        future_returns = future_returns.loc[common_index]
        rank_ic_list = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            for dt in factor_values.index:
                daily_data = pd.concat([factor_values.loc[dt], future_returns.loc[dt]], axis=1)
                daily_data.columns = ['factor', 'return']
                daily_data_cleaned = daily_data.dropna()
                if len(daily_data_cleaned) < 2: continue
                if daily_data_cleaned['factor'].nunique() < 2 or daily_data_cleaned['return'].nunique() < 2:
                    continue
                corr, _ = spearmanr(daily_data_cleaned['factor'], daily_data_cleaned['return'])
                if not np.isnan(corr): rank_ic_list.append(corr)
        if not rank_ic_list: return 0.0, 0.0
        return np.mean(rank_ic_list), np.std(rank_ic_list)

    @staticmethod
    def calculate_turnover(factor_values: pd.DataFrame, top_n: int = 50):
        turnover_list = []
        for i in range(1, len(factor_values)):
            prev_top = factor_values.iloc[i - 1].nlargest(top_n).index
            curr_top = factor_values.iloc[i].nlargest(top_n).index
            turnover = 1.0 - len(set(prev_top) & set(curr_top)) / top_n
            turnover_list.append(turnover)
        return np.mean(turnover_list) if turnover_list else 0.0

    def run_backtest(self, formula_str: str, value_name: str = "factor"):
        factor_values = self.calculate_factor(formula_str)
        if factor_values is None:
            return None

        rank_ic_mean, rank_ic_std = self.calculate_rank_ic(factor_values, self.future_returns)
        icir = (rank_ic_mean / rank_ic_std) if rank_ic_std > 0 else 0.0
        turnover = self.calculate_turnover(factor_values)

        return {"rank_ic_mean": rank_ic_mean, "rank_ic_std": rank_ic_std, "icir": icir, "turnover": turnover}