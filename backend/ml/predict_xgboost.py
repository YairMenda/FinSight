import sys
import yfinance as yf
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def predict_stock(symbol: str, days_to_predict: int, from_date: str):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=3 * 365)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    from_date_dt = pd.to_datetime(from_date)

    data = yf.download(symbol, start=start_str, end=end_str, auto_adjust=False)['Adj Close']
    data = pd.DataFrame(data)
    data.columns = [symbol]

    for window in [2, 5, 10, 20, 60]:
        data[f'{symbol}_rolling_{window}'] = data[symbol].rolling(window=window).mean().ffill().bfill()
        data[f'{symbol}_rollingSTD_{window}'] = data[symbol].rolling(window=window).std().ffill().bfill()
        data[f'{symbol}_rollingMedian_{window}'] = data[symbol].rolling(window=window).median().ffill().bfill()

    data.bfill(inplace=True)
    data.ffill(inplace=True)

    scaler_X = StandardScaler()
    X_scaled = pd.DataFrame(scaler_X.fit_transform(data.drop(columns=[symbol])),
                            columns=data.columns.drop(symbol),
                            index=data.index)

    y = data[symbol]

    X_train, X_test, y_train, y_test, train_dates, test_dates = train_test_split(
        X_scaled, y, X_scaled.index, test_size=0.2, random_state=42, shuffle=False)

    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    X_last_n = X_scaled.iloc[-days_to_predict:].copy()
    y_future_pred = model.predict(X_last_n)

    test_dates = pd.Series(test_dates)
    future_dates = pd.date_range(start=test_dates.iloc[-1] + pd.Timedelta(days=1), periods=days_to_predict, freq='D')

    # === Filter JSON data based on from_date ===
    plot_data = []

    for date, actual, predicted in zip(test_dates, y_test, y_pred):
        if pd.to_datetime(date) >= from_date_dt:
            plot_data.append({
                "date": pd.to_datetime(date).strftime('%Y-%m-%d'),
                "actual": float(actual),
                "predicted": float(predicted)
            })

    for date, forecast in zip(future_dates, y_future_pred):
        if pd.to_datetime(date) >= from_date_dt:
            plot_data.append({
                "date": pd.to_datetime(date).strftime('%Y-%m-%d'),
                "forecast": float(forecast)
            })

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "std": float(np.std(y_pred))
    }

    return {
        "symbol": symbol,
        "plot_data": plot_data,
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": y_pred,
        "test_dates": test_dates,
        "future_dates": future_dates,
        "y_future_pred": y_future_pred
    }


def plot_predictions(result):
    symbol = result["symbol"]
    y_test = result["y_test"]
    y_pred = result["y_pred"]
    test_dates = result["test_dates"]
    future_dates = result["future_dates"]
    y_future_pred = result["y_future_pred"]
    metrics = result["metrics"]

    plt.figure(figsize=(14, 8))
    plt.plot(test_dates, y_test.values, label='Actual (y_test)', marker='o')
    plt.plot(test_dates, y_pred, label='Predicted (y_pred)', marker='x')
    plt.plot(future_dates, y_future_pred, label=f'Forecast (next {len(y_future_pred)} days)', marker='^')
    plt.title(f"Stock: {symbol} — Actual vs Predicted vs Forecasted")
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)

    metric_text = (f"R²: {metrics['r2']:.4f}\n"
                   f"MAE: {metrics['mae']:.4f}\n"
                   f"MSE: {metrics['mse']:.4f}\n"
                   f"STD of Predictions: {metrics['std']:.4f}")
    plt.gcf().text(0.15, 0.75, metric_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


# # === Example Run ===
# result = predict_stock("AAPL", 30, from_date="2024-06-01")
# plot_predictions(result)
#
# # === Print Filtered JSON Data ===
# print(json.dumps(result["plot_data"], indent=2))

if __name__ == "__main__":
    try:
        symbol = sys.argv[1]
        future_days = int(sys.argv[2])
        from_date = sys.argv[3]

        result = predict_stock(symbol, future_days, from_date)
        print(json.dumps(result["plot_data"]))
        # plot_predictions(result)

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
