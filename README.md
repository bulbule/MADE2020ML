# made_hw_4_ml_2020
We are delighted to present you the service which provides financial advice for investor. Take advantage of it and soon we'll see you on the Forbes list :blush:
The service allows to realize the following operations:
1)    Stock price prediction to the next five days using Gaussian process
What should I do?
1.    Select **a start date** in the calendar.
2.    The model will predict the stock price to the next five days and build the graph which demonstrates the real and predicted stock price.
How does it work?
The model is based on Gaussian Process for regression from GPY library. The model is trained on data of 60 previous days. We use the data of the following companies: Boeing, Cisco and Intel. All data were taken from quandl.com.
2)    Simulation of daily trading during 2017
What should I do?
1.    Select **a final date** in the calendar.
2.    Enter the amount of your bet.
3.    The model will imitate the daily traiding and build the graph which demonstrates the real and predicted stock price. At the same time the model will return your potential revenue.
How does it work?
The model is based on LSTM from keras library. The model is trained on data of four previous years. We use the data of the following companies: Boeing, Cisco and Intel. All data were taken from quandl.com.
