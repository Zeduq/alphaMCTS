import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import RandomForestClassifier
from tradelearn.strategy.backtest import Strategy


class RandomForestStrategy(Strategy):

    bt_begin_date = '2020-01-01'
    bt_end_date = '2022-06-22'
    fea_list = []
    model_dict = {}

    def init(self):

        data = self.data.df.swaplevel(0, 1, axis=1).stack().reset_index(level=1)

        if not self.model_dict:

            for date in pd.date_range(start=self.bt_begin_date, end=self.bt_end_date, freq='12MS'):
                bt_train_data = data.query(
                    f"date >= '{date - relativedelta(months=12 * 3)}' and date < '{date}'").iloc[:300]
                bt_x_train, bt_y_train = bt_train_data[self.fea_list], bt_train_data['label']

                model = RandomForestClassifier(random_state=42, n_jobs=-1)
                model.fit(bt_x_train, bt_y_train)
                self.model_dict[date.year] = model

        ind_df = pd.DataFrame({'date': data.index.unique()}).set_index('date')
        for date in pd.date_range(start=self.bt_begin_date, end=self.bt_end_date, freq='12MS'):
            for symbol in data['code'].unique():
                print(symbol)
                bt_x_test = data.query(
                    f"code == '{symbol}' and date >= '{date}' and date < '{date + relativedelta(months=12 * 1)}'")
                if bt_x_test.empty:
                    cur_ind_df = ind_df.query(f"date >= '{date}' and date < '{date + relativedelta(months=12 * 1)}'")
                    pre_proba = pd.DataFrame([0]*len(cur_ind_df), columns=[symbol], index=cur_ind_df.index)
                else:
                    bt_x_test = bt_x_test[self.fea_list]
                    pre_proba = pd.DataFrame(self.model_dict[date.year].predict_proba(bt_x_test)[:, 1],
                                    columns=[symbol], index=bt_x_test.index)
                ind_df = ind_df.combine_first(pre_proba)

        self.proba = self.I(ind_df, overlay=False)

    def next(self):

        print(self.proba.df.index[-1])

        if self.data.index[-1] < pd.to_datetime(self.bt_begin_date):
            return

        # 重置投资组合的持仓权重
        self.alloc.assume_zero()

        # 得到当天的各标的预测概率
        proba = self.proba.df.iloc[-1]

        # 根据概率指标进行标的数量的筛选，同时设置持仓权重分配
        bucket = self.alloc.bucket['equity']
        bucket.append(proba.sort_values(ascending=False))
        bucket.trim(limit=3)
        bucket.weight_explicitly(weight=1 / 3)
        bucket.apply(method='update')

        # 更新投资组合持仓权重
        self.rebalance(cash_reserve=0.1)