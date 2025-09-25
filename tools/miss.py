import numpy as np
import pandas as pd


class Miss:

    def __init__(self):
        pass

    @staticmethod
    def replace_nan_indu(data, ind):
        data = pd.merge(data, ind, how='inner', on=['code'])
        industry_mean_df = data.groupby('ind_code').mean().apply(lambda x: x.fillna(x.mean()), axis=0)

        for index in data.index:
            for col in data.columns.drop('ind_code'):
                if np.isnan(data.loc[index, col]):
                    data.loc[index, col] = industry_mean_df.loc[data.loc[index, 'ind_code'], col]
        data = data.drop(columns=['ind_code'])
        return data

    @staticmethod
    def drop_nan_columns(data, threshold=0.5):
        data = data.pivot_table(index='date', columns='code', dropna=False).swaplevel(0, 1, axis=1)
        return data.dropna(axis=1, thresh=threshold*len(data))