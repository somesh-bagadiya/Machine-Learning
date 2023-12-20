import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import math
from sklearn.metrics import accuracy_score
import time

start_time = time.time()

# Load data
df = pd.read_csv('stock_price.csv')
data = df['Close']

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(np.array(data).reshape(-1, 1))

# Train-test split
training_size = int(len(data) * 0.70)
test_size = len(data) - training_size
train_data, test_data = data[0:training_size, :], data[training_size:len(data), :1]


# Create dataset for time series
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        temp = dataset[i:(i + time_step), 0]
        dataX.append(temp)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# Define time step
time_step = 100

# Create training and testing sets
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape data for LSTM model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Create binary labels (1 for up, 0 for down)
X_train = X_train[1:]
X_test = X_test[1:]

y_train_binary = np.where(y_train[1:] > y_train[:-1], 1, 0)
y_test_binary = np.where(y_test[1:] > y_test[:-1], 1, 0)

# Build LSTM model for binary classification
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(2, activation='softmax'))  # Output layer for binary classification
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X_train, y_train_binary, validation_data=(X_test, y_test_binary), epochs=100, verbose=1)

training_time = time.time() - start_time
print("Training Time:", training_time)

# Evaluate the model
accuracy = model.evaluate(X_test, y_test_binary)[1]
print("Accuracy:", accuracy)

# Make predictions
test_predictions_binary = np.argmax(model.predict(X_test), axis=1)

# Inverse transform predictions
test_predict = scaler.inverse_transform(y_test.reshape(-1, 1))
test_predict = np.concatenate([np.full(time_step, np.nan), test_predict.flatten()])

# Plotting
plt.figure(figsize=(12, 6))

print(len(test_predict[time_step:]), len(data)+1, len(range(time_step, len(data)+1)))

print(len(test_predictions_binary), len(y_test_binary))

# # Plot actual stock prices
# plt.subplot(2, 1, 1)
# plt.plot(scaler.inverse_transform(data), label="Actual Stock Price")
# plt.scatter(range(0, len(X_test)+1), test_predict[time_step:], c=test_predictions_binary, cmap='coolwarm', marker='x', label="Predicted Direction")
# plt.title('Actual Stock Prices and Predicted Directions')
# plt.legend()

# Plot training and validation loss
plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
