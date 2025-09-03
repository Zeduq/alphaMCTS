

class Outlier:

    def __init__(self):
        pass

    @staticmethod
    def winorize_med(data, scale, axis=0):
        def func(col):
            med = col.median()
            med1 = abs(col - med).median()
            col[col > med + scale * med1] = med + scale * med1
            col[col < med - scale * med1] = med - scale * med1
            return col

        win_factor_data = data.apply(func, axis=axis)
        return win_factor_data
