import sys
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, BatchNormalization
import numpy as np
import json
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta


def GRU_model_predict(symbol, days_to_forecast, from_date):
    from_date = pd.to_datetime(from_date)
    end_date = datetime.datetime.now()
    start_date = end_date - timedelta(days=6 * 365)

    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if data.empty:
            raise ValueError(f"No data available for symbol '{symbol}'.")
    except Exception as e:
        raise RuntimeError(f"Failed to download data for '{symbol}': {str(e)}")

    data = data[['Close', 'High', 'Low', 'Open', 'Volume']]
    data['Adj Close'] = data['Close']
    data = data.dropna()

    data['High_Low_Diff'] = data['High'] - data['Low']
    data['Open_Close_Diff'] = data['Open'] - data['Close']
    data['Adj_Close_5d_Rolling'] = data['Adj Close'].rolling(window=5).mean()
    data['Adj_Close_10d_Rolling'] = data['Adj Close'].rolling(window=10).mean()
    data = data.dropna()

    X = data.drop('Adj Close', axis=1)
    y = data['Adj Close']

    scaler_X, scaler_y = RobustScaler(), RobustScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    split_index = int(0.9 * len(X_scaled))
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y_scaled[:split_index], y_scaled[split_index:]

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        GRU(98, return_sequences=True),
        Dropout(0.1),
        GRU(48, return_sequences=True),
        Dropout(0.1),
        GRU(22, return_sequences=False),
        Dropout(0.1),
        BatchNormalization(),
        Dense(10),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)

    y_pred = model.predict(X_test)
    y_pred_inverse = scaler_y.inverse_transform(y_pred)
    y_test_inverse = scaler_y.inverse_transform(y_test)
    y_train_inverse = scaler_y.inverse_transform(y_train)

    pct_change_distribution = pd.Series(y_pred_inverse.flatten()).pct_change().dropna()
    future_predictions = [y_pred_inverse[-1]]
    for _ in range(days_to_forecast):
        pct = np.random.choice(pct_change_distribution)
        future_predictions.append(future_predictions[-1] * (1 + pct))

    last_data_date = data.index[-1]
    train_dates = data.index[:split_index]
    test_dates = data.index[split_index:]
    forecast_dates = pd.date_range(last_data_date + timedelta(days=1), periods=days_to_forecast)

    all_data = {
        "train": [{"day": d.strftime("%Y-%m-%d"), "price": float(p)}
                  for d, p in zip(train_dates, y_train_inverse.flatten())],
        "test_actual": [{"day": d.strftime("%Y-%m-%d"), "price": float(p)}
                        for d, p in zip(test_dates, y_test_inverse.flatten())],
        "test_predicted": [{"day": d.strftime("%Y-%m-%d"), "price": float(p)}
                           for d, p in zip(test_dates, y_pred_inverse.flatten())],
        "forecast": [{"day": d.strftime("%Y-%m-%d"), "price": float(p)}
                     for d, p in zip(forecast_dates, future_predictions[1:])]
    }

    filtered_result = {
        key: [item for item in all_data[key] if pd.to_datetime(item["day"]) >= from_date]
        for key in all_data
    }

    return filtered_result


def plot_results(result_json):
    plt.figure(figsize=(14, 6))

    if result_json.get("train"):
        train_dates = [datetime.datetime.strptime(item["day"], "%Y-%m-%d") for item in result_json["train"]]
        train_prices = [item["price"] for item in result_json["train"]]
        plt.plot(train_dates, train_prices, label="Train")

    if result_json.get("test_actual"):
        test_dates = [datetime.datetime.strptime(item["day"], "%Y-%m-%d") for item in result_json["test_actual"]]
        test_prices = [item["price"] for item in result_json["test_actual"]]
        plt.plot(test_dates, test_prices, label="Test Actual")

    if result_json.get("test_predicted"):
        pred_dates = [datetime.datetime.strptime(item["day"], "%Y-%m-%d") for item in result_json["test_predicted"]]
        pred_prices = [item["price"] for item in result_json["test_predicted"]]
        plt.plot(pred_dates, pred_prices, label="Test Predicted")

    if result_json.get("forecast"):
        forecast_dates = [datetime.datetime.strptime(item["day"], "%Y-%m-%d") for item in result_json["forecast"]]
        forecast_prices = [item["price"] for item in result_json["forecast"]]
        plt.plot(forecast_dates, forecast_prices, '--', label="Forecast", color="orange")

    plt.title("Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        symbol = sys.argv[1]
        future_days = int(sys.argv[2])
        from_date = sys.argv[3]

        result_json = GRU_model_predict(symbol, future_days, from_date)
        sys.stdout.write(json.dumps(result_json))

    except Exception as e:
        sys.stdout.write(json.dumps({"error": str(e)}))
        sys.exit(1)
