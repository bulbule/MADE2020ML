# MADE_hw4_ml_2020
Team assignment for hw4 of mail.ru MADE ML course. Build a model of your choice and present a demo. 
We created a toy service which predicts price.
The service allows a user to realize the following operations:
1. Stock price prediction for the next five days using a Gaussian process

	1.Select **a start date** in the calendar.
	
	2.The model will predict the stock price to the next five days and build the graph which demonstrates the real and predicted stock price.

How does it work?
The model is based on Gaussian Process for regression from GPy library. The model is trained on data for 60 previous days. We use the data of the following companies: Boeing, Cisco and Intel.

2. Simulation of daily trading during 2017

	1.Select **a final date** in the calendar.
	
	2.Enter the amount of your bet.
	
The model will imitate the daily traiding and build the graph which demonstrates the real and predicted stock price.
At the same time the model will return your potential revenue.

How does it work?
The model is based on LSTM from keras library. The model is trained on data of four previous years. We use the data of the following companies: Boeing, Cisco and Intel. All data were taken from [quandl.com](https://www.quandl.com)

To run a server: `python index.py`
