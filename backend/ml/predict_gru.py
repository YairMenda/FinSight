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
    # Convert from_date to datetime
    from_date = pd.to_datetime(from_date)

    # Calculate date range (last 6 years + buffer for forecasting)
    end_date = datetime.datetime.now()
    start_date = end_date - timedelta(days=6 * 365)  # 6 years of training data

    try:
        print(f"Downloading data for: {symbol}")
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if data.empty:
            print(f"No data available for {symbol}")
            return {}
    except Exception as e:
        print(f"Failed to download data for {symbol}: {e}")
        return {}

    # Prepare data
    data = data[['Close', 'High', 'Low', 'Open', 'Volume']]
    data['Adj Close'] = data['Close']
    data = data.dropna()

    # Add basic features
    data['High_Low_Diff'] = data['High'] - data['Low']
    data['Open_Close_Diff'] = data['Open'] - data['Close']
    data['Adj_Close_5d_Rolling'] = data['Adj Close'].rolling(window=5).mean()
    data['Adj_Close_10d_Rolling'] = data['Adj Close'].rolling(window=10).mean()
    data = data.dropna()

    # Scale data
    X = data.drop('Adj Close', axis=1)
    y = data['Adj Close']
    scaler_X, scaler_y = RobustScaler(), RobustScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    # Split data (90% train, 10% test)
    split_index = int(0.9 * len(X_scaled))
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y_scaled[:split_index], y_scaled[split_index:]

    # Build and train model
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

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_inverse = scaler_y.inverse_transform(y_pred)
    y_test_inverse = scaler_y.inverse_transform(y_test)
    y_train_inverse = scaler_y.inverse_transform(y_train)

    # Generate forecast
    pct_change_distribution = pd.Series(y_pred_inverse.flatten()).pct_change().dropna()
    future_predictions = [y_pred_inverse[-1]]
    for _ in range(days_to_forecast):
        pct = np.random.choice(pct_change_distribution)
        future_predictions.append(future_predictions[-1] * (1 + pct))

    # Create date ranges
    last_data_date = data.index[-1]
    train_dates = data.index[:split_index]
    test_dates = data.index[split_index:]
    forecast_dates = pd.date_range(last_data_date + timedelta(days=1), periods=days_to_forecast)

    # Prepare all data with dates
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

    # Filter to only include dates after from_date
    filtered_result = {}
    for key in all_data:
        filtered_result[key] = [
            item for item in all_data[key]
            if pd.to_datetime(item["day"]) >= from_date
        ]

    return filtered_result


def plot_results(result_json):
    plt.figure(figsize=(14, 6))

    # Plot training data if available
    if result_json.get("train"):
        train_dates = [datetime.datetime.strptime(item["day"], "%Y-%m-%d") for item in result_json["train"]]
        train_prices = [item["price"] for item in result_json["train"]]
        plt.plot(train_dates, train_prices, label="Train")

    # Plot test actual if available
    if result_json.get("test_actual"):
        test_dates = [datetime.datetime.strptime(item["day"], "%Y-%m-%d") for item in result_json["test_actual"]]
        test_prices = [item["price"] for item in result_json["test_actual"]]
        plt.plot(test_dates, test_prices, label="Test Actual")

    # Plot test predicted if available
    if result_json.get("test_predicted"):
        pred_dates = [datetime.datetime.strptime(item["day"], "%Y-%m-%d") for item in result_json["test_predicted"]]
        pred_prices = [item["price"] for item in result_json["test_predicted"]]
        plt.plot(pred_dates, pred_prices, label="Test Predicted")

    # Plot forecast if available
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


# # Example usage
# if __name__ == "__main__":
#     # Get prediction results as JSON (only including data after 2021-01-01)
#     result_json = GRU_model("AAPL", 30, "2021-01-01")
#
#     # Save JSON to file
#     with open("stock_prediction_results.json", "w") as f:
#         json.dump(result_json, f, indent=2)
#     print("Results saved to stock_prediction_results.json")
#
#     # Plot from JSON
#     plot_results(result_json)

if __name__ == "__main__":
    try:
        # Read exactly three args
        symbol = sys.argv[1]
        future_days = int(sys.argv[2])
        from_date = sys.argv[3]

        # Run your model
        result_json = GRU_model_predict(symbol, future_days, from_date)

        # Print _only_ the JSON we need
        sys.stdout.write(json.dumps(result_json))

    except Exception as e:
        # If anything goes wrong, return it as JSON
        sys.stdout.write(json.dumps({"error": str(e)}))
        sys.exit(1)
