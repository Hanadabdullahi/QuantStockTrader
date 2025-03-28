import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Honestly fick denna kod från chat, vet inte riktigt vad den göra men aja.
ticker = 'BTC-USD'
start_date = '2015-01-01'
end_date = '2023-12-31'  # Adjust the end date as needed
window_size = 30  # Number of past days used as input features
lookahead = 7     # Days ahead to predict the trend


data = yf.download(ticker, start=start_date, end=end_date)

# Create a target column: 1 if price in 7 days is higher than current price, else 0
data['Target'] = np.where(data['Close'].shift(-lookahead) > data['Close'], 1, 0)

# Drop the last few rows with NaN target values due to shifting
data.dropna(inplace=True)


features = []
labels = []
for i in range(window_size, len(data) - lookahead):
    feature_seq = data['Close'].iloc[i - window_size:i].values  # last `window_size` closing prices
    label = data['Target'].iloc[i]  # label corresponding to the current day
    features.append(feature_seq)
    labels.append(label)

features = np.array(features)
labels = np.array(labels)

# Reshape features for LSTM [samples, time steps, features]
features = features.reshape(features.shape[0], features.shape[1], 1)


n_samples, n_timesteps, n_features = features.shape
features_reshaped = features.reshape(n_samples, n_timesteps)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_reshaped)
features_scaled = features_scaled.reshape(n_samples, n_timesteps, n_features)


X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, shuffle=False)

#Enkel lstm modell
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(window_size, 1)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
