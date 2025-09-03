import os
import warnings
import numpy as np
import pandas as pd
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

warnings.filterwarnings('ignore')

if __name__ == '__main__':

    data_path = './res/data/000300SH_POST.csv'
    if not os.path.exists(data_path):
        print('start to generate new processed data!')

        begin_date = '2017-01-01'
        end_date = '2023-06-26'
        raw_data = Query.read_csv(data_path='./data/000300SH.csv', begin=begin_date, end=end_date).iloc[:, 1:]

        alpha101_data_file = './res/data/alpha101_data.csv'
        if not os.path.exists(alpha101_data_file):
            s_data = raw_data.pivot(index='date', columns='code')

            alpha101_file = './data/alpha101.csv'
            if not os.path.exists(alpha101_file):
                s_df = Query.alphas101(s_data)
                s_df.to_csv(alpha101_file, encoding='utf_8_sig')
            else:
                s_df = pd.read_csv(alpha101_file, parse_dates=['date'], dtype={'code': str},
                                   low_memory=True, encoding='utf_8_sig').iloc[:, 1:]
            raw_data = pd.merge(raw_data, s_df, how='left', on=['date', 'code'])
            del s_df

            raw_data.to_csv(alpha101_data_file, encoding='utf_8_sig')
        else:
            raw_data = pd.read_csv(alpha101_data_file, parse_dates=['date'], dtype={'code': str},
                                    low_memory=True, encoding='utf_8_sig').iloc[:, 1:]

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

        fina_data.to_csv(data_path, encoding='utf_8_sig')

    else:
        fina_data = pd.read_csv(data_path, index_col=0, parse_dates=['date'], dtype={'code': str}, low_memory=True, encoding='utf_8_sig')

    fina_data = fina_data.fillna(0)  # 这里已经对缺失值进行了处理，不需要再进行处理

    alpha101_compared_file = './res/data/alpha101_factor_compare.csv'
    if not os.path.exists(alpha101_compared_file):
        eval_res = Examine.factor_compare(fina_data, 'ind_code', 'cir_a')
        eval_res.to_csv(alpha101_compared_file, encoding='utf_8_sig')
    else:
        eval_res = pd.read_csv(alpha101_compared_file, encoding='utf_8_sig')
        eval_res = eval_res.sort_values(by='risk_adjusted_ic(ir)', ascending=False)
    fea_list = eval_res['name'][:20].tolist()  # 取前二十因子作为特征

    bt_begin_date = '2020-01-01'
    bt_end_date = '2022-06-21'
    bt_test_df = fina_data.query(f"date >= '2017-01-01' and date < '2022-06-21'")
    bt_test_df = bt_test_df.pivot_table(index='date', columns='code', dropna=False).swaplevel(0, 1, axis=1)
    bt_test_df = bt_test_df.sort_values(by='code', axis=1).fillna(method='ffill')
    hs300 = Query.read_csv('./data/hs300_index.csv', bt_begin_date, bt_end_date).set_index('date')

    bt = Backtest(bt_test_df, RandomForestStrategy, cash=1000000, commission=.0002, margin=1, trade_on_close=True)
    stats = bt.run(bt_begin_date=bt_begin_date, bt_end_date=bt_end_date, fea_list=fea_list)
    stats.to_csv(f'./res/{bt_begin_date}_{bt_end_date}_stats.csv')
    bt.plot(filename='./res/backtest.html', plot_volume=True, superimpose=True, plot_drawdown=True, plot_return=True, plot_allocation=True)

    Evaluate.analysis_report(stats, hs300, engine='quantstats', filename='./res/report.html')



