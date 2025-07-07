import sys
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, BatchNormalization
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Use Agg backend which doesn't require GUI
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta


def GRU_model_predict(symbol, days_to_forecast, from_date):
    from_date = pd.to_datetime(from_date)
    end_date = datetime.datetime.now()
    start_date = end_date - timedelta(days=6 * 365)

    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if data.empty:
        raise ValueError(f"No data available for symbol '{symbol}'.")

    data = data[['Close', 'High', 'Low', 'Open', 'Volume']]
    data['Adj Close'] = data['Close']
    data = data.dropna()

    # Feature engineering
    data['High_Low_Diff'] = data['High'] - data['Low']
    data['Open_Close_Diff'] = data['Open'] - data['Close']
    data['Adj_Close_5d_Rolling'] = data['Adj Close'].rolling(window=5).mean()
    data['Adj_Close_10d_Rolling'] = data['Adj Close'].rolling(window=10).mean()
    data = data.dropna()

    X = data.drop('Adj Close', axis=1)
    y = data['Adj Close']

    # Scaling
    scaler_X, scaler_y = RobustScaler(), RobustScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    # Train/test split
    split_index = int(0.9 * len(X_scaled))
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y_scaled[:split_index], y_scaled[split_index:]

    # GRU Model
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

    # Predict on all historical data (not just test)
    y_all_pred_scaled = model.predict(X_scaled)
    y_all_pred = scaler_y.inverse_transform(y_all_pred_scaled)
    y_all_true = scaler_y.inverse_transform(y_scaled)

    # Evaluation metrics on test data
    y_test_pred = model.predict(X_test)
    y_test_pred = scaler_y.inverse_transform(y_test_pred)
    y_test_true = scaler_y.inverse_transform(y_test)

    mae = mean_absolute_error(y_test_true, y_test_pred)
    mse = mean_squared_error(y_test_true, y_test_pred)
    std = float(np.std(y_test_pred))
    r2 = r2_score(y_test_true, y_test_pred)

    # Forecast future prices based on simulated pct changes
    pct_change_distribution = pd.Series(y_all_pred.flatten()).pct_change().dropna()
    last_price = y_all_pred[-1][0]
    future_predictions = [last_price]
    for _ in range(days_to_forecast):
        pct = np.random.choice(pct_change_distribution)
        next_price = future_predictions[-1] * (1 + pct)
        future_predictions.append(next_price)

    # Build response
    all_dates = data.index
    actual_data = []
    predicted_data = []

    for date, true_price, pred_price in zip(all_dates, y_all_true.flatten(), y_all_pred.flatten()):
        if date >= from_date:
            actual_data.append([date.strftime("%Y-%m-%d"), float(true_price)])
            predicted_data.append([date.strftime("%Y-%m-%d"), float(pred_price)])

    forecast_dates = pd.date_range(data.index[-1] + timedelta(days=1), periods=days_to_forecast)
    forecasted_data = [[date.strftime("%Y-%m-%d"), float(price)] for date, price in zip(forecast_dates, future_predictions[1:])]

    return {
        "symbol": symbol,
        "actual": actual_data,
        "predicted": predicted_data,
        "forecasted": forecasted_data,
        "metrics": {
            "MAE": float(mae),
            "MSE": float(mse),
            "STD": float(std),
            "R2": float(r2)
        }
    }


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
