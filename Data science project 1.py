#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tensorflow --upgrade')

import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, History
import math
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
import numpy as np
import tensorflow as tf
import numpy as np
import tensorflow as tf
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import random


from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [10, 7.5]
get_ipython().system('pip install pmdarima')

import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
import itertools


# In[ ]:


np.random.seed(42)
tf.random.set_seed(42)


# In[ ]:


# fetching the SPDR S&P 500 ETF data from Yahoo Finance form the period of Janauary 2019 to May 2024
ticker = 'SPY'  # Ticker for SPDR S&P 500 ETF Trust
sp500_data = yf.download(ticker, start="2019-07-01", end="2024-06-30")
print(sp500_data)


# In[ ]:


# Extracting the Closing price column only which we will be using for our prediction
data = sp500_data[['Close']]
print(data.head())


# Exploring the stock data to get insights about the data
plt.figure(figsize=(12, 6))
plt.plot(data['Close'])


# In[ ]:


# performing descriptive statistics
data.describe()


# In[ ]:


# Plotting histogram to undertand the distribution of the data

plt.hist(data['Close'])
plt.title('Distribution of S&P 500 Stock Price 2019-2024')
plt.xlabel('Stock Price')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[ ]:


# Performing Feature Engineering on the data to create a year and month columns for data exploration

data.loc[:, 'Year'] = data.index.year
data.loc[:, 'Month'] = data.index.strftime('%b')
data.loc[:, 'Day'] = data.index.strftime('%a')


# In[ ]:


plt.figure(figsize=(8, 4))
sns.boxplot(data=data, x='Year', y='Close')
plt.title('S&P 500 Closing Prices by Year (2019-2024)')
plt.xlabel('Year')
plt.ylabel('Closing Price')
plt.show()


# In[ ]:




