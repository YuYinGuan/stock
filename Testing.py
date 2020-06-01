import math
import pandas_datareader as web
import numpy as np
import pandas as pd
#from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential
#from keras.layers import Dense, LSTM
import matplotlib.pyplot as plit
plt.style.use('fivethirtyeight')

import yfinance as yf  
 
# Get the data for the stock Apple by specifying the stock ticker, start date, and end date
data = yf.download('AAPL','2016-01-01','2018-01-01')
 
# Plot the close prices
import matplotlib.pyplot as plt
data.Close.plot()
plt.show()
