import pandas as pd
from gplearn.genetic import SymbolicTransformer


def generate_gp_formulas():
    print("加载数据中...")
    df = pd.read_csv('D:/AAProject/Data/000300SH.csv', parse_dates=['date']).dropna()
    df = df[df['date'] < '2019-01-01']

    # 准备特征 (X) 和 目标 (y, 未来1天收益率)
    features = ['open', 'high', 'low', 'close', 'volume', 'vwap']
    X = df[features].values

    # 计算未来收益率作为拟合目标
    df['target'] = df.groupby('code')['close'].shift(-1) / df['close'] - 1
    y = df['target'].fillna(0).values

    print("正在运行遗传规划 (GP) 繁衍因子...")
    # 配置遗传规划引擎 (生成 50 个特征)
    gp = SymbolicTransformer(
        generations=3,  # 迭代代数
        population_size=1000,  # 种群大小
        hall_of_fame=50,  # 最终保留的优秀公式数量
        n_components=50,  # 我们需要提取 50 个公式
        function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min'),
        feature_names=features,
        random_state=42,
        n_jobs=-1
    )

    gp.fit(X, y)

    print("\n✅ GP 挖掘完成！获得的 50 个黑盒公式如下：")
    gp_formulas = []
    for i, program in enumerate(gp):
        formula_str = str(program)
        gp_formulas.append(formula_str)
        print(f"GP公式 {i + 1}: {formula_str}")

    return gp_formulas


if __name__ == "__main__":
    formulas = generate_gp_formulas()