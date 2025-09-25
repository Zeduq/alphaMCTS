

class Scale:

    def __init__(self):
        pass

    @staticmethod
    def standardlize(data):
        data = (data-data.mean())/data.std()
        return data
