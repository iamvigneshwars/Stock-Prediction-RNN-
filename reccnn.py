#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 20:52:29 2019

@author: imvigneshwars
"""


#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

#feature scaling

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

#creaing a data structure with 60 timesteps and 1 output

x_train = []
y_train = []

for i in range(60, 1258):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

x_train = np.array(x_train)

y_train = np.array(y_train)

#reshaping the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#builging the RNN

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

regressor = Sequential()

# adding the first lstm layer and dropout regularization

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))
# adding the first lstm layer and dropout regularization

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# adding the second lstm layer and dropout regularization

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# adding the fourth lstm layer and dropout regularization

regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(0.2))

# adding the output layers

regressor.add(Dense(units = 1))

# compiling the rnn
regressor.compile(optimizer ='adam', loss = 'mean_squared_error')

# fitting the RNN to the training set

regressor.fit(x_train, y_train, epochs = 100 , batch_size = 64)

regressor.save('model.h5')

# Making the predictions and visualising the results
# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()




















