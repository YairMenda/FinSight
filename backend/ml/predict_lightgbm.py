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


def lightGBM_predict(symbol, from_date_str, days_to_predict):
    end_date_for_training = datetime.now()
    start_date_for_training = end_date_for_training - timedelta(days=6 * 365)

    data = yf.download(symbol, start=start_date_for_training.strftime('%Y-%m-%d'),
                       end=end_date_for_training.strftime('%Y-%m-%d'), auto_adjust=False)

    if data.empty:
        raise ValueError(f"No data downloaded for {symbol}. Check ticker/date range.")

    data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns.values]

    adj_close_col_name = f'Adj Close_{symbol}'
    volume_col_name = f'Volume_{symbol}'
    open_col_name = f'Open_{symbol}'
    high_col_name = f'High_{symbol}'
    low_col_name = f'Low_{symbol}'
    close_col_name = f'Close_{symbol}'

    if adj_close_col_name not in data.columns:
        adj_close_col_name = 'Adj Close'
        volume_col_name = 'Volume'
        open_col_name = 'Open'
        high_col_name = 'High'
        low_col_name = 'Low'
        close_col_name = 'Close'

    data['Target'] = data[adj_close_col_name]

    def create_features(df, adj_close_col, volume_col):
        df_copy = df.copy()
        adj_close_series = df_copy[adj_close_col]
        volume_series = df_copy[volume_col]

        for window in [2, 5, 10, 20, 60]:
            df_copy[f'rolling_mean_{window}'] = adj_close_series.rolling(window=window).mean()
            df_copy[f'rolling_std_{window}'] = adj_close_series.rolling(window=window).std()
            df_copy[f'rolling_median_{window}'] = adj_close_series.rolling(window=window).median()

        for window in [12, 26, 50, 100]:
            df_copy[f'EMA_{window}'] = ta.trend.ema_indicator(adj_close_series, window=window)

        df_copy['RSI'] = ta.momentum.rsi(adj_close_series, window=14)
        df_copy['MACD'] = ta.trend.macd(adj_close_series)
        df_copy['MACD_Signal'] = ta.trend.macd_signal(adj_close_series)
        df_copy['MACD_Diff'] = ta.trend.macd_diff(adj_close_series)
        df_copy['BB_High'] = ta.volatility.bollinger_hband(adj_close_series)
        df_copy['BB_Low'] = ta.volatility.bollinger_lband(adj_close_series)
        df_copy['BB_Mid'] = ta.volatility.bollinger_mavg(adj_close_series)
        df_copy['OBV'] = ta.volume.on_balance_volume(adj_close_series, volume_series)

        for lag in [1, 2, 3, 5, 10]:
            df_copy[f'Adj_Close_Lag_{lag}'] = adj_close_series.shift(lag)

        df_copy['Daily_Return'] = adj_close_series.pct_change()

        df_copy['Day_of_Week'] = df_copy.index.dayofweek
        df_copy['Day_of_Year'] = df_copy.index.dayofyear
        df_copy['Month'] = df_copy.index.month

        return df_copy

    data_with_features = create_features(data, adj_close_col_name, volume_col_name)
    data_with_features.fillna(method='ffill', inplace=True)
    data_with_features.fillna(method='bfill', inplace=True)
    data_with_features.dropna(inplace=True)

    if data_with_features.empty:
        raise RuntimeError("Data became empty after feature engineering and NaN handling.")

    original_price_cols = [open_col_name, high_col_name, low_col_name, close_col_name,
                           adj_close_col_name, volume_col_name]
    features_to_drop_final = [col for col in original_price_cols if col in data_with_features.columns]
    features_to_drop_final.append('Target')

    X = data_with_features.drop(columns=features_to_drop_final)
    y = data_with_features['Target']

    if len(X) != len(y):
        raise ValueError("Mismatch between features (X) and target (y).")

    # Split data for training and testing
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=7,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        colsample_bytree=0.8,
        subsample=0.8
    )

    model.fit(X_train_scaled, y_train)

    # Make predictions on test data
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    std = np.std(y_pred)
    r2 = r2_score(y_test, y_pred)

    historical_daily_returns = data_with_features['Daily_Return'].dropna()
    mean_daily_return = historical_daily_returns.mean()
    std_daily_return = historical_daily_returns.std()

    historical_volume = data_with_features[volume_col_name].dropna()
    mean_volume = historical_volume.mean()
    std_volume = historical_volume.std()

    data_with_features['Daily_Range_Ratio'] = (data_with_features[high_col_name] - data_with_features[low_col_name]) / data_with_features[close_col_name]
    data_with_features['Open_Close_Ratio'] = (data_with_features[open_col_name] - data_with_features[close_col_name]) / data_with_features[close_col_name]
    data_with_features['Close_AdjClose_Ratio'] = data_with_features[close_col_name] / data_with_features[adj_close_col_name]

    mean_daily_range_ratio = data_with_features['Daily_Range_Ratio'].mean()
    std_daily_range_ratio = data_with_features['Daily_Range_Ratio'].std()
    mean_open_close_ratio = data_with_features['Open_Close_Ratio'].mean()
    std_open_close_ratio = data_with_features['Open_Close_Ratio'].std()
    mean_close_adjclose_ratio = data_with_features['Close_AdjClose_Ratio'].mean()

    last_historical_date = y.index[-1]
    last_historical_adj_close = y.iloc[-1]
    last_historical_close = data[close_col_name].iloc[-1]

    max_window = max([2, 5, 10, 12, 14, 20, 26, 50, 60, 100]) + 10
    historical_window = data.iloc[-(max_window + 1):].copy()

    forecast_dates, predictions = [], []
    current_adj_close = last_historical_adj_close
    current_close = last_historical_close

    for i in range(1, days_to_predict + 1):
        pred_date = last_historical_date + pd.Timedelta(days=i)

        simulated_return = np.random.normal(mean_daily_return, std_daily_return)
        simulated_adj_close = current_adj_close * (1 + simulated_return)
        simulated_volume = max(0, np.random.normal(mean_volume, std_volume))
        simulated_close = simulated_adj_close / mean_close_adjclose_ratio
        simulated_open = simulated_close + np.random.normal(mean_open_close_ratio, std_open_close_ratio) * simulated_close
        simulated_range = abs(np.random.normal(mean_daily_range_ratio, std_daily_range_ratio)) * simulated_close
        simulated_high = max(simulated_open, simulated_close) + simulated_range * np.random.uniform(0, 0.5)
        simulated_low = min(simulated_open, simulated_close) - simulated_range * np.random.uniform(0, 0.5)

        new_day = pd.DataFrame(index=[pred_date])
        new_day[adj_close_col_name] = simulated_adj_close
        new_day[volume_col_name] = simulated_volume
        new_day[open_col_name] = simulated_open
        new_day[high_col_name] = simulated_high
        new_day[low_col_name] = simulated_low
        new_day[close_col_name] = simulated_close

        temp_df = pd.concat([historical_window, new_day])
        features = create_features(temp_df, adj_close_col_name, volume_col_name)
        features.fillna(method='ffill', inplace=True)
        features.fillna(method='bfill', inplace=True)

        X_future = features.iloc[-1:].drop(columns=[col for col in features_to_drop_final if col in features.columns], errors='ignore')

        for col in (set(X.columns) - set(X_future.columns)):
            X_future[col] = 0
        X_future = X_future[X.columns]

        X_future.dropna(inplace=True)
        if X_future.empty:
            raise RuntimeError(f"Missing features for {pred_date}, cannot continue prediction.")

        scaled_future = scaler_X.transform(X_future)
        pred_price = model.predict(scaled_future)[0]

        forecast_dates.append(pred_date)
        predictions.append(pred_price)

        current_adj_close = pred_price
        current_close = simulated_close

        new_hist = pd.DataFrame(index=[pred_date])
        new_hist[adj_close_col_name] = pred_price
        new_hist[volume_col_name] = simulated_volume
        new_hist[open_col_name] = simulated_open
        new_hist[high_col_name] = simulated_high
        new_hist[low_col_name] = simulated_low
        new_hist[close_col_name] = simulated_close

        historical_window = pd.concat([historical_window, new_hist]).iloc[-(max_window + 1):]

    # Filter data from the specified date
    from_date_dt = datetime.strptime(from_date_str, "%Y-%m-%d")
    
    # Prepare actual data (test set actual values)
    actual_data = []
    for date, price in y_test.items():
        if date >= from_date_dt:
            actual_data.append([date.strftime("%Y-%m-%d"), float(price)])
    
    # Prepare predicted data (test set predictions)
    predicted_data = []
    for date, price in zip(y_test.index, y_pred):
        if date >= from_date_dt:
            predicted_data.append([date.strftime("%Y-%m-%d"), float(price)])
    
    # Prepare forecasted data (future predictions)
    forecasted_data = []
    for date, price in zip(forecast_dates, predictions):
        forecasted_data.append([date.strftime("%Y-%m-%d"), float(price)])

    result = {
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
