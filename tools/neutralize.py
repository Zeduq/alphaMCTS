import numpy as np
import pandas as pd
import statsmodels.api as sm


class Neutralize:

    def __init__(self):
        pass

    @staticmethod
    def neutralize(data, ind, cir):
        for col in data.columns:
            x = pd.concat([np.log(cir),
                           pd.get_dummies(ind, drop_first=True, columns=[ind.columns[0]])], axis=1)
            y = data[col]
            model = sm.OLS(y.astype(float), x.astype(float)).fit()
            data[col] = model.resid
        return data
