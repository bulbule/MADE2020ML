import pandas as pd
import GPy


def get_prediction(date_start: str,
                   stock: pd.DataFrame,
                   period: int):
    def get_days(days_index):
        days = days_index.values.copy()
        for i, d in enumerate(days_index.values):
            if i != len(days) - 1:
                if days[i + 1] - days[i] < 0:
                    days[i + 1:] = days[i] + days[i + 1:]
        return days
    date_start = pd.to_datetime(date_start)
    train_dates_start = date_start - pd.to_timedelta(60, unit='d')
    if date_start.dayofweek == 0:
        train_dates_end = date_start - pd.to_timedelta(3, unit='d')
        predict_dates_end = date_start + pd.to_timedelta(period - 1, unit='d')
    else:
        train_dates_end = date_start - pd.to_timedelta(1, unit='d')
        predict_dates_end = date_start + pd.to_timedelta(period + 2 - 1, unit='d')
    train = stock.loc[train_dates_start: train_dates_end].copy()
    days_index = train.index.day
    days = get_days(days_index)
    x = days
    y = train['Open'].values
    kernel = GPy.kern.Linear(1) * GPy.kern.Matern32(1, lengthscale=.5, variance=1) + \
             GPy.kern.White(input_dim=1) + \
             GPy.kern.Matern52(1, lengthscale=.5, variance=5)
    kernel[".*white"].constrain_fixed(0.06)
    # kernel['.*white'].constrain_fixed(1e-2)
    gp = GPy.models.GPRegression(x.reshape(-1, 1), y.reshape(-1, 1),
                                 kernel,
                                 normalizer=True)
    gp.optimize()

    test = stock.loc[train_dates_start: predict_dates_end].copy()
    days_index = test.index.day
    days_test = get_days(days_index)
    x_test = days_test
    y_pred, y_std = gp.predict(x_test.reshape(-1, 1))
    return train_dates_start, test, y_pred, y_std
