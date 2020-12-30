DESCRIPTION = '''We are delighted to present you the service which provides financial advice for investor. Take advantage of it and soon we'll see you on the Forbes list!\n
The service allows a user to realize the following operations:\n
1)    **Stock price prediction for the next five days using Gaussian process**\n
*What should I do?*\n
1.    Select a start date in the calendar.\n
2.    The model will predict the stock price for the next five days and build the graph which demonstrates the real and predicted stock price.\n
How does it work?\n
The model is based on Gaussian Process for regression from GPY library. The model is trained on data of 60 previous days.\n
2)    **Simulation of daily trading during 2017**\n
*What should I do?*\n
1.    Select a final date in the calendar.\n
2.    Enter the amount of your bet.\n
3.    The model will imitate the daily trading and build the graph which demonstrates the real and predicted stock price. At the same time the model will return your potential revenue.\n
How does it work?\n
The model is based on LSTM from keras library. The model is trained on data of three previous years.\n
We use the data of the following companies: Boeing, Cisco and Intel. All data was taken from [quandl.com](https://www.quandl.com).'''
