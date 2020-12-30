from typing import Dict, List, Tuple

from tensorflow.keras.models import load_model
import joblib
import numpy as np
import pandas as pd

FEATURE_SIZE_CONST = 60
ROOT_DIR = 'models_and_data/'
MODELS = [ROOT_DIR + 'model_BA.h5',
          ROOT_DIR + 'model_CSCO.h5',
          ROOT_DIR + 'model_INTC.h5']
START_PRICES = [ROOT_DIR + 'start_BA.csv',
                ROOT_DIR + 'start_CSCO.csv',
                ROOT_DIR + 'start_INTC.csv']
SCALERS = [ROOT_DIR + 'scaler_BA.pkl',
           ROOT_DIR + 'scaler_CSCO.pkl',
           ROOT_DIR + 'scaler_INTC.pkl']
STOCKS_NAMES = ['BA', 'CSCO', "INTC"]
TRUE_PRICES = [ROOT_DIR + 'true_BA_2017.csv',
               ROOT_DIR + 'true_CSCO_2017.csv',
               ROOT_DIR + 'true_INTC_2017.csv']


class Stock:
    def __init__(self,
                 model_file: str,
                 scaler_file: str,
                 start_prices_file: str,
                 true_prices_file: str
                 ) -> None:
        self.model = load_model(model_file)
        self.scaler = joblib.load(scaler_file)
        self.predicted_prices: pd.DataFrame = pd.read_csv(start_prices_file, parse_dates=['date'])
        self.true_prices: pd.DataFrame = pd.read_csv(true_prices_file, parse_dates=['date'])
        self.true_prices = pd.concat([self.predicted_prices, self.true_prices], ignore_index=True)

    def predict_price_for_the_next_day(self, date: pd._libs.tslibs.timestamps.Timestamp):
        x = self._get_some_days_before_the_day(date)
        x_sc = self.scaler.transform(x.reshape(-1, 1))
        x_sc = np.reshape(x_sc, (1, x_sc.shape[0], 1))
        next_price_sc = self.model.predict(x_sc)
        next_price = self.scaler.inverse_transform(next_price_sc)[0][0]
        self.predicted_prices = self.predicted_prices.append(pd.Series({'date': date, 'high': next_price}),
                                                             ignore_index=True)
        return next_price

    def get_last_price(self) -> float:
        return self.predicted_prices.iloc[-1]['high']

    def get_predicted_price(self, date) -> float:
        return self.predicted_prices.loc[self.predicted_prices['date'] == date]['high'].values[0]

    def get_true_price(self, date) -> float:
        return self.true_prices.loc[self.true_prices['date'] == date]['high'].values[0]

    def _get_some_days_before_the_day(self, date: pd._libs.tslibs.timestamps.Timestamp):
        return self.true_prices[self.true_prices['date'] < date].tail(FEATURE_SIZE_CONST).high.values


class User_Wallet:
    def __init__(self, cash: float, user_stocks: Dict[str, int]):
        self.cash = cash
        self.stocks = user_stocks

    def sell_stock(self, stock_name: str, current_price: float):
        stock_count = self.stocks[stock_name]
        self.cash += current_price * stock_count
        self.stocks[stock_name] -= stock_count

    def buy_stock(self, stock_name: str, current_price: float):
        count_we_can_afford = self.cash // current_price
        self.cash -= current_price * count_we_can_afford
        self.stocks[stock_name] += count_we_can_afford

    def get_cash_in_wallet(self):
        return self.cash

    def get_cost_of_wallet(self,
                           date: pd._libs.tslibs.timestamps.Timestamp,
                           stocks: Dict[str, Stock]
                           ) -> float:
        ans = self.cash
        for stock_name in stocks.keys():
            ans += self.stocks[stock_name] * stocks[stock_name].get_true_price(date)
        return ans


def build_list_of_trading_days() -> List[pd._libs.tslibs.timestamps.Timestamp]:
    true_data = pd.read_csv(TRUE_PRICES[0])
    dates_2017 = true_data['date'].values
    dates_2017 = list(map(pd._libs.tslibs.timestamps.Timestamp, dates_2017))
    return dates_2017


def one_day_action(
        wallet: User_Wallet,
        stocks: Dict[str, Stock],
        date: pd._libs.tslibs.timestamps.Timestamp,
) -> Tuple[Dict[str, Stock], User_Wallet]:
    for stock in STOCKS_NAMES:
        current_price = stocks[stock].get_true_price(date)
        predicted_price = stocks[stock].predict_price_for_the_next_day(date)
        if predicted_price < current_price:
            wallet.sell_stock(stock, current_price)
        else:
            wallet.buy_stock(stock, current_price)
    return stocks, wallet


DEFAULT_STOCKS = dict()
for stock_name in STOCKS_NAMES:
    DEFAULT_STOCKS[stock_name] = 0


def full_period(
        cash: float,
        final_day: pd._libs.tslibs.timestamps.Timestamp,
        user_stocks: Dict[str, int] = DEFAULT_STOCKS
) -> Tuple[float,
           List[pd._libs.tslibs.timestamps.Timestamp],
           Dict[str, List[float]],
           Dict[str, List[float]],
           List[float]]:
    dates_2017 = build_list_of_trading_days()
    wallet = User_Wallet(cash, user_stocks)
    stocks = {}
    dates, wallet_cost = [], []
    true_prices, predicted_prices = {}, {}
    for i, stock_name in enumerate(STOCKS_NAMES):
        stocks[stock_name] = Stock(MODELS[i], SCALERS[i], START_PRICES[i], TRUE_PRICES[i])
        true_prices[stock_name] = []
        predicted_prices[stock_name] = []
    for date in dates_2017:
        if date <= final_day:
            stocks, wallet = one_day_action(wallet, stocks, date)
            dates.append(date)
            wallet_cost.append(wallet.get_cost_of_wallet(date, stocks))
            for stock_name in STOCKS_NAMES:
                true_prices[stock_name].append(stocks[stock_name].get_true_price(date))
                predicted_prices[stock_name].append(stocks[stock_name].get_predicted_price(date))
        else:
            break
    for stock_name in STOCKS_NAMES:
        current_price = stocks[stock_name].get_true_price(date)
        wallet.sell_stock(stock_name, current_price)
    return wallet.get_cash_in_wallet(), dates, true_prices, predicted_prices, wallet_cost


if __name__ == '__main__':
    final_date = pd.Timestamp('2017-12-31T12')
    user_stocks = {}
    for stock_name in STOCKS_NAMES:
        user_stocks[stock_name] = 0
    cash, dates, true_prices, predicted_prices, wallet_cost = full_period(1000, final_date, user_stocks)
    print(f'Cash on {final_date} is {cash}')
    print(f'Trading days in the peroid was {dates}')
    for stock_name in STOCKS_NAMES:
        print(f'True prices of {stock_name} in the period were {true_prices[stock_name]}')
        print(f'Predicted prices of {stock_name} in the period were {predicted_prices[stock_name]}')
    print(f'Costs of the wallet was the following {wallet_cost}')