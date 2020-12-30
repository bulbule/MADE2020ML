# MADE_hw4_ml_2020
Team assignment for hw4 of mail.ru MADE ML course. Build a model of your choice and present a demo. 
We created a toy service which predicts price.
The service allows a user to realize the following operations:
<ol>
<li>Stock price prediction for the next five days using a Gaussian process</li>
<ol>
	<li> Select **a start date** in the calendar.</li>
	<li>  The model will predict the stock price to the next five days and build the graph which demonstrates the real and predicted stock price.</li>
</ol>
How does it work?
The model is based on Gaussian Process for regression from GPy library. The model is trained on data for 60 previous days. We use the data of the following companies: Boeing, Cisco and Intel.
<li>Simulation of daily trading during 2017</li>
<ol>
<li>Select **a final date** in the calendar.</li>
<li>Enter the amount of your bet.</li>
<li>The model will imitate the daily traiding and build the graph which demonstrates the real and predicted stock price.
    At the same time the model will return your potential revenue.</li>
</ol>
How does it work?
The model is based on LSTM from keras library. The model is trained on data of four previous years. We use the data of the following companies: Boeing, Cisco and Intel. All data were taken from [quandl.com](https://www.quandl.com)
