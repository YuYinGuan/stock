
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
#from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential
#from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import datetime as dt

plt.style.use('fivethirtyeight')

datestart = dt.datetime(2019,1,1)
dateend = dt.datetime(2020,5,25)
df = web.DataReader("AAPL", data_source='yahoo', start=datestart, end=dateend)

df.shape

plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)

plt.show()
