# import sys
# import json
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime
# from xgboost import XGBRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#
# WINDOW_SIZE = 15
#
#
# def build_features(series, window=WINDOW_SIZE):
#     X, y = [], []
#     for i in range(len(series) - window):
#         X.append(series[i:i + window])
#         y.append(series[i + window])
#     return np.array(X), np.array(y)
#
#
# def run(symbol, data, end_date, future_days=7):
#     if symbol not in data.columns:
#         raise ValueError(f"Symbol {symbol} not found in provided data.")
#
#     df_symbol = data[symbol].dropna()
#     series = df_symbol.values
#
#     # Create aligned date index for samples (X/y)
#     dates = df_symbol.index[WINDOW_SIZE:]
#
#     if len(series) <= WINDOW_SIZE:
#         raise ValueError(f"Not enough data to build features (length={len(series)}, window={WINDOW_SIZE})")
#
#     # Build time-series features
#     X, y = build_features(series)
#
#     # Scale X
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # Split chronologically
#     X_train, X_test, y_train, y_test, train_dates, test_dates = train_test_split(
#         X_scaled, y, dates, test_size=0.2, shuffle=False, random_state=42
#     )
#
#     # Train model
#     model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
#     model.fit(X_train, y_train)
#
#     # Predict and evaluate
#     y_pred = model.predict(X_test)
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#
#     # Predict next N days using recursive window
#     last_window = series[-WINDOW_SIZE:].tolist()
#     future_preds = []
#     for _ in range(future_days):
#         window_scaled = scaler.transform([last_window])
#         pred = model.predict(window_scaled)[0]
#         future_preds.append(pred)
#         last_window.pop(0)
#         last_window.append(pred)
#
#     # Build prediction dates
#     base = pd.to_datetime(end_date)
#     prediction_dates = [(base + pd.Timedelta(days=i + 1)).strftime('%Y-%m-%d') for i in range(future_days)]
#
#     # Rebuild historical values for output
#     df_symbol = data[symbol].dropna()
#     df_symbol.index = pd.to_datetime(df_symbol.index, errors='coerce')
#
#     return {
#         "symbol": symbol,
#         "r2": float(r2),
#         "mae": float(mae),
#         "mse": float(mse),
#         "history": [
#             {"date": idx.strftime('%Y-%m-%d'), "close": round(val, 2)}
#             for idx, val in df_symbol.items() if not pd.isnull(idx)
#         ],
#         "predictions": [
#             {"date": d, "price": round(float(p), 2)}
#             for d, p in zip(prediction_dates, future_preds)
#         ],
#
#         "y_test": y_test.tolist(),
#         "y_pred": y_pred.tolist(),
#         "test_dates": [d.strftime('%Y-%m-%d') for d in test_dates],
#         "y_future_pred": [float(p) for p in future_preds]
#     }
#
#
# def plot_results(symbol, test_dates, y_test, y_pred, y_future_pred, future_days, r2, mae, mse):
#     std_pred = np.std(y_pred)
#
#     # Convert test_dates if needed
#     test_dates = pd.to_datetime(test_dates)
#
#     if len(test_dates) == 0:
#         print("❌ Cannot plot: test_dates is empty.")
#         return
#
#     # Ensure future forecast dates
#     try:
#         future_dates = pd.date_range(start=test_dates[-1] + pd.Timedelta(days=1), periods=future_days, freq='D')
#     except Exception as e:
#         print(f"❌ Failed to create future_dates: {e}")
#         return
#
#     # Begin plotting
#     plt.figure(figsize=(14, 8))
#
#     plt.plot(test_dates, y_test, label='Actual (y_test)', marker='o')
#     plt.plot(test_dates, y_pred, label='Predicted (y_pred)', marker='x')
#     plt.plot(future_dates, y_future_pred, label='Forecast (future)', marker='^')
#
#     plt.title(f"Stock: {symbol} — Actual vs Predicted vs Forecasted")
#     plt.xlabel('Date')
#     plt.ylabel('Stock Price')
#     plt.legend()
#     plt.grid(True)
#
#     metric_text = (
#         f"R²: {r2:.4f}\n"
#         f"MAE: {mae:.4f}\n"
#         f"MSE: {mse:.4f}\n"
#         f"STD of Predictions: {std_pred:.4f}"
#     )
#     plt.gcf().text(0.15, 0.75, metric_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
#
#     plt.tight_layout()
#     plt.show()
#
#
# if __name__ == "__main__":
#     try:
#         symbol = sys.argv[1]
#         date_from_obj = datetime.strptime(sys.argv[2], '%Y-%m-%d')
#         date_to_obj = datetime.strptime(sys.argv[3], '%Y-%m-%d')
#         date_from = date_from_obj.strftime('%Y-%m-%d')
#         date_to = date_to_obj.strftime('%Y-%m-%d')
#         future_days = int(sys.argv[4]) if len(sys.argv) > 4 else 7
#
#         tickers = [symbol]
#         data = yf.download(tickers, start=date_from, end=date_to, auto_adjust=False)['Adj Close']
#
#         result = run(symbol, data, date_to, future_days)
#         # print(json.dumps(result))
#
#         plot_results(
#             symbol,
#             result["test_dates"],
#             result["y_test"],
#             result["y_pred"],
#             result["y_future_pred"],
#             future_days,
#             result["r2"],
#             result["mae"],
#             result["mse"]
#         )
#
#     except Exception as e:
#         print(json.dumps({"error": str(e)}))
#         sys.exit(1)
import sys
import json
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

WINDOW_SIZE = 15

def build_features(series, window=WINDOW_SIZE):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    return np.array(X), np.array(y)

def run(symbol, df_symbol, end_date, future_days=7):
    series = df_symbol.values
    if len(series) <= WINDOW_SIZE:
        raise ValueError("Not enough data to build features")

    dates = df_symbol.index[WINDOW_SIZE:]
    X, y = build_features(series)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test, train_dates, test_dates = train_test_split(
        X_scaled, y, dates, test_size=0.2, shuffle=False, random_state=42
    )

    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Future prediction
    last_window = series[-WINDOW_SIZE:].tolist()
    future_preds = []
    for _ in range(future_days):
        window_scaled = scaler.transform([last_window])
        pred = model.predict(window_scaled)[0]
        future_preds.append(pred)
        last_window.pop(0)
        last_window.append(pred)

    base = pd.to_datetime(end_date)
    prediction_dates = [(base + timedelta(days=i + 1)).strftime('%Y-%m-%d') for i in range(future_days)]

    return {
        "symbol": symbol,
        "r2": float(r2),
        "mae": float(mae),
        "mse": float(mse),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "test_dates": [d.strftime('%Y-%m-%d') for d in test_dates],
        "y_future_pred": [float(p) for p in future_preds],
        "prediction_dates": prediction_dates
    }

def plot_results(symbol, test_dates, y_test, y_pred, future_dates, y_future_pred, r2, mae, mse):
    test_dates = pd.to_datetime(test_dates)
    future_dates = pd.to_datetime(future_dates)

    plt.figure(figsize=(14, 8))
    plt.plot(test_dates, y_test, label='Actual Prices', marker='o')
    plt.plot(test_dates, y_pred, label='Predicted Prices', marker='x')
    plt.plot(future_dates, y_future_pred, label='Future Forecast', marker='^')

    plt.title(f"{symbol} — GRU Forecast Results")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    metrics = f"R²: {r2:.4f}\nMAE: {mae:.2f}\nMSE: {mse:.2f}"
    plt.gcf().text(0.15, 0.75, metrics, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        symbol = sys.argv[1]
        from_date = datetime.strptime(sys.argv[2], '%Y-%m-%d')
        future_days = int(sys.argv[3]) if len(sys.argv) > 3 else 7

        start_date = (from_date - timedelta(days=365 * 3)).strftime('%Y-%m-%d')
        end_date = from_date.strftime('%Y-%m-%d')

        df = yf.download(symbol, start=start_date, end=end_date)
        if df.empty or 'Adj Close' not in df.columns:
            raise ValueError("No data found for symbol")

        df_symbol = df['Adj Close'].dropna()
        df_symbol.index = pd.to_datetime(df_symbol.index)

        result = run(symbol, df_symbol, end_date, future_days)

        # הדפסה כ־JSON לפלט במערכת backend
        print(json.dumps(result))

        # שרטוט לבדיקה ידנית
        plot_results(
            result["symbol"],
            result["test_dates"],
            result["y_test"],
            result["y_pred"],
            result["prediction_dates"],
            result["y_future_pred"],
            result["r2"],
            result["mae"],
            result["mse"]
        )

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
