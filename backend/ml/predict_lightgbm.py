import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import matplotlib
matplotlib.use('Agg')  # Use Agg backend which doesn't require GUI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import ta
from datetime import datetime, timedelta
import json

warnings.filterwarnings('ignore')


def lightGBM_predict(symbol, from_date_str, days_to_forecast):
    from_date = pd.to_datetime(from_date_str)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=6 * 365)

    # Download data
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if data.empty:
        raise ValueError(f"No data available for symbol '{symbol}'.")

    data = data[['Close', 'High', 'Low', 'Open', 'Volume']]
    data['Adj Close'] = data['Close']
    data.dropna(inplace=True)

    # Feature engineering
    data['High_Low_Diff'] = data['High'] - data['Low']
    data['Open_Close_Diff'] = data['Open'] - data['Close']
    data['Adj_Close_5d_Rolling'] = data['Adj Close'].rolling(window=5).mean()
    data['Adj_Close_10d_Rolling'] = data['Adj Close'].rolling(window=10).mean()
    data.dropna(inplace=True)

    X = data.drop(columns=['Adj Close'])
    y = data['Adj Close']

    # Scaling
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Train/test split
    split_index = int(0.9 * len(X_scaled))
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # LightGBM Model
    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=7,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict full history
    y_all_pred = model.predict(X_scaled)

    # Evaluate on test
    y_test_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    std = float(np.std(y_test_pred))
    r2 = r2_score(y_test, y_test_pred)

    # Forecast future prices based on simulated pct changes
    pct_change_distribution = pd.Series(y_all_pred).pct_change().dropna()
    last_price = y_all_pred[-1]
    future_predictions = [last_price]
    for _ in range(days_to_forecast):
        pct = np.random.choice(pct_change_distribution)
        next_price = future_predictions[-1] * (1 + pct)
        future_predictions.append(next_price)

    # Build response
    all_dates = data.index
    actual_data = []
    predicted_data = []

    for date, true_price, pred_price in zip(all_dates, y.values, y_all_pred):
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


    return result


def plot_forecast(forecast_json_str):
    forecast_data = json.loads(forecast_json_str)

    ticker = forecast_data['symbol']
    actual_data = forecast_data['actual']
    forecast_data_points = forecast_data['forecasted']
    forecast_stats = forecast_data['metrics']
    last_historical_date_str = actual_data[-1][0]

    actual_dates = [pd.to_datetime(dp[0]) for dp in actual_data]
    actual_prices = [dp[1] for dp in actual_data]

    forecast_dates = [pd.to_datetime(dp[0]) for dp in forecast_data_points]
    forecast_prices = [dp[1] for dp in forecast_data_points]

    last_historical_date = pd.to_datetime(last_historical_date_str)

    plt.figure(figsize=(18, 9))

    plt.plot(actual_dates, actual_prices, label='Actual Prices', color='blue', linewidth=2)
    if forecast_prices:
        plt.plot(forecast_dates, forecast_prices, label=f'Forecast ({len(forecast_prices)} Days)',
                 color='green', linestyle='--', marker='^', markersize=4)
        plt.axvline(x=last_historical_date, color='grey', linestyle='--', label='Forecast Start')

    plt.title(f"{ticker} Stock Price Forecast", fontsize=18)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    plt.legend()
    plt.grid(True)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()


# --- Example usage ---
if __name__ == "__main__":
    SYMBOL = 'TSLA'
    FROM_DATE = '2024-01-01'
    DAYS = 30

    try:
        output = lightGBM_predict(SYMBOL, FROM_DATE, DAYS)
        plot_forecast(json.dumps(output))
    except Exception as e:
        print(f"Error: {e}")
