import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
import time

start_time = time.time()

df = pd.read_csv('stock_price.csv')
data = df['Close'] 

scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(np.array(data).reshape(-1, 1))

training_size = int(len(data) * 0.70)
test_size = len(data) - training_size
train_data, test_data = data[0:training_size, :], data[training_size:len(data), :1]

print(len(train_data), len(test_data))


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        temp = dataset[i:(i + time_step), 0]
        dataX.append(temp)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print(X_train.shape), print(y_train.shape)
print(X_test.shape), print(y_test.shape)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')  # Stochastic Gradient Descent
model.summary()
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=1)

training_time = time.time() - start_time
print("Training Time:", training_time)

start_time = time.time()
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

predicting_time = time.time() - start_time
print("Predicting Time:", predicting_time)

loss = model.evaluate(X_test, y_test)

a = math.sqrt(mean_squared_error(y_train, train_predict))
b = math.sqrt(mean_squared_error(y_test, test_predict))
print("Train Error", a,"Test Error:", b)

look_back = 100
trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

testPredictPlot = np.empty_like(data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(data) - 1, :] = test_predict

# plot baseline and predictions
plt.ylabel('Stock Price ($)')
plt.xlabel('Days')
plt.title('Stock Prices Over Last 5 Years')

plt.plot(scaler.inverse_transform(data), label="Actual Stock Price")
plt.plot(trainPredictPlot, label="Training ")
plt.plot(testPredictPlot, label="Predicted")
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
