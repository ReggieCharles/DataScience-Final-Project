#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Prepocessing the data so it can be suitable for Deep learning
scaler = MinMaxScaler()
scaled_data =scaler.fit_transform(data2['Close'].values.reshape(-1,1))

# splitting the data into training and testing
train_size = int(len(scaled_data) * 0.80) # size of training data is 80% of the total data
test_size = len(scaled_data) - train_size # size of testing data which will be 20%

#train_data, test_data = scaled_data[:train_size,:], scaled_data[train_size:,:]
train_data,test_data = scaled_data[0:train_size,:],scaled_data[train_size:len(scaled_data),:]
train_data.shape, test_data.shape

scaled_data


# In[ ]:


#LSTM Model LSTM Model deals with sequence of data in prediction. 
#That is the model learns to predict the next values in the sequence based on past data. 
#As a result, the model wont have an explict target column.
#hence, we will have to create the sequnce in which the model will use past data to predict target values 
#as shown in the next line of code. We will employ the use of timeseriesgenerator for this

# number of previous time steps(sequence of time) to use before making prediction on the next one
steps=30
​
#creating a sequence of data that uses 30 past data(X) to make prediction(y)
train_generator= TimeseriesGenerator(train_data,train_data, length=steps, batch_size=20)
test_generator= TimeseriesGenerator(test_data, test_data, length=steps, batch_size=2)


# In[ ]:


np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)



# Building a Sequential LSTM Model
lstm_model = Sequential()
lstm_model.add(LSTM(units=30, return_sequences=True, input_shape=(30,1)))
lstm_model.add(LSTM(units=60, return_sequences=True))
lstm_model.add(LSTM(units=60))
lstm_model.add(Dense(1))

# compiling the model
lstm_model.compile(optimizer='adam',loss='mean_squared_error', metrics=['mean_absolute_error'])

# Training the model on the training data and assessing  its performance on the validation data
lstm_model.fit(train_generator, epochs=100, batch_size=20)


# In[ ]:


# Display the model summary
lstm_model.summary()

# Plotting the model
plot_model(lstm_model)


# In[ ]:


# Predicting on the test data
lstm_predictions = lstm_model.predict(test_generator)

