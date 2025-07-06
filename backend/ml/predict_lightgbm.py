import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import ta
from datetime import datetime, timedelta
import json

warnings.filterwarnings('ignore')


def lightGBM_predict(symbol, from_date_str, days_to_predict):
    # Define the full historical period for model training (last 6 years)
    end_date_for_training = datetime.now()
    start_date_for_training = end_date_for_training - timedelta(days=6 * 365)  # Approximately 6 years

    print(
        f"Downloading data for {symbol} from {start_date_for_training.strftime('%Y-%m-%d')} to {end_date_for_training.strftime('%Y-%m-%d')} for training...")
    data = yf.download(symbol, start=start_date_for_training.strftime('%Y-%m-%d'),
                       end=end_date_for_training.strftime('%Y-%m-%d'), auto_adjust=False)

    if data.empty:
        print(f"Error: No data downloaded for {symbol}. Check ticker/date range.")
        return None

    # Column Name Handling
    data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns.values]

    # Identify relevant column names, with fallback for non-MultiIndex data
    adj_close_col_name = f'Adj Close_{symbol}'
    volume_col_name = f'Volume_{symbol}'
    open_col_name = f'Open_{symbol}'
    high_col_name = f'High_{symbol}'
    low_col_name = f'Low_{symbol}'
    close_col_name = f'Close_{symbol}'

    if adj_close_col_name not in data.columns:  # Fallback for non-MultiIndex
        adj_close_col_name = 'Adj Close'
        volume_col_name = 'Volume'
        open_col_name = 'Open'
        high_col_col_name = 'High'
        low_col_name = 'Low'
        close_col_name = 'Close'

    data['Target'] = data[adj_close_col_name]

    # --- Feature Engineering Function ---
    def create_features(df, adj_close_col, volume_col):
        df_copy = df.copy()
        adj_close_series = df_copy[adj_close_col]
        volume_series = df_copy[volume_col]

        # Rolling Window Features
        for window in [2, 5, 10, 20, 60]:
            df_copy[f'rolling_mean_{window}'] = adj_close_series.rolling(window=window).mean()
            df_copy[f'rolling_std_{window}'] = adj_close_series.rolling(window=window).std()
            df_copy[f'rolling_median_{window}'] = adj_close_series.rolling(window=window).median()

        # Technical Analysis Indicators (TA Library)
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

        # Lagged Price Features
        for lag in [1, 2, 3, 5, 10]:
            df_copy[f'Adj_Close_Lag_{lag}'] = adj_close_series.shift(lag)

        # Daily Returns
        df_copy['Daily_Return'] = adj_close_series.pct_change()

        # Time-based Features
        df_copy['Day_of_Week'] = df_copy.index.dayofweek
        df_copy['Day_of_Year'] = df_copy.index.dayofyear
        df_copy['Month'] = df_copy.index.month

        return df_copy

    print("Generating features for historical data...")
    data_with_features = create_features(data, adj_close_col_name, volume_col_name)
    data_with_features.fillna(method='ffill', inplace=True)
    data_with_features.fillna(method='bfill', inplace=True)
    data_with_features.dropna(inplace=True)

    if data_with_features.empty:
        print("Error: Data became empty after feature engineering and NaN handling.")
        return None

    # --- Prepare Data for Model Training ---
    original_price_cols = [open_col_name, high_col_name, low_col_name,
                           close_col_name, adj_close_col_name, volume_col_name]
    features_to_drop_final = [col for col in original_price_cols if col in data_with_features.columns]
    features_to_drop_final.append('Target')

    X = data_with_features.drop(columns=features_to_drop_final)
    y = data_with_features['Target']

    if len(X) != len(y):
        print("Error: Mismatch in number of samples between features (X) and target (y).")
        return None

    X_train_full = X
    y_train_full = y

    # Calculate historical statistics for future feature simulation
    historical_daily_returns = data_with_features['Daily_Return'].dropna()
    mean_daily_return = historical_daily_returns.mean()
    std_daily_return = historical_daily_returns.std()

    historical_volume = data_with_features[volume_col_name].dropna()
    mean_volume = historical_volume.mean()
    std_volume = historical_volume.std()

    data_with_features['Daily_Range_Ratio'] = (data_with_features[high_col_name] - data_with_features[low_col_name]) / \
                                              data_with_features[close_col_name]
    data_with_features['Open_Close_Ratio'] = (data_with_features[open_col_name] - data_with_features[close_col_name]) / \
                                             data_with_features[close_col_name]
    data_with_features['Close_AdjClose_Ratio'] = data_with_features[close_col_name] / data_with_features[
        adj_close_col_name]

    mean_daily_range_ratio = data_with_features['Daily_Range_Ratio'].dropna().mean()
    std_daily_range_ratio = data_with_features['Daily_Range_Ratio'].dropna().std()
    mean_open_close_ratio = data_with_features['Open_Close_Ratio'].dropna().mean()
    std_open_close_ratio = data_with_features['Open_Close_Ratio'].dropna().std()
    mean_close_adjclose_ratio = data_with_features['Close_AdjClose_Ratio'].dropna().mean()

    # Scale features (X) for training
    scaler_X = StandardScaler()
    X_train_full_scaled = scaler_X.fit_transform(X_train_full)
    X_train_full_scaled = pd.DataFrame(X_train_full_scaled, columns=X_train_full.columns, index=X_train_full.index)

    # --- Train LightGBM Model ---
    print("Training LightGBM model...")
    model = LGBMRegressor(n_estimators=500,
                          learning_rate=0.03,
                          max_depth=7,
                          num_leaves=31,
                          min_child_samples=20,
                          random_state=42,
                          n_jobs=-1,
                          colsample_bytree=0.8,
                          subsample=0.8)

    model.fit(X_train_full_scaled, y_train_full)
    print("Model training complete.")

    # --- Recursive Multi-Step Forecasting ---
    print(f"Generating recursive {days_to_predict}-day forecast with simulated inputs...")

    last_historical_date = y.index[-1]
    last_historical_adj_close = y.iloc[-1]
    last_historical_volume = data[volume_col_name].iloc[-1]
    last_historical_close = data[close_col_name].iloc[-1]

    max_feature_window = max([2, 5, 10, 12, 14, 20, 26, 50, 60, 100]) + 10

    historical_window_for_forecast = data.iloc[-(max_feature_window + 1):].copy()

    future_predictions = []
    forecast_dates_list = []  # Renamed to avoid conflict with plotting function's internal variable

    current_predicted_adj_close = last_historical_adj_close
    current_predicted_close = last_historical_close

    for i in range(1, days_to_predict + 1):
        current_date_to_predict = last_historical_date + pd.Timedelta(days=i)

        # Simulate daily return, volume, and OHL prices based on historical distributions
        simulated_daily_return = np.random.normal(mean_daily_return, std_daily_return)
        simulated_next_adj_close = current_predicted_adj_close * (1 + simulated_daily_return)
        simulated_volume = max(0, np.random.normal(mean_volume, std_volume))

        simulated_close = simulated_next_adj_close / mean_close_adjclose_ratio
        simulated_open = simulated_close + np.random.normal(mean_open_close_ratio,
                                                            std_open_close_ratio) * simulated_close

        simulated_daily_range = np.abs(
            np.random.normal(mean_daily_range_ratio, std_daily_range_ratio)) * simulated_close
        simulated_high = max(simulated_open, simulated_close) + simulated_daily_range * np.random.uniform(0, 0.5)
        simulated_low = min(simulated_open, simulated_close) - simulated_daily_range * np.random.uniform(0, 0.5)

        # Create a new DataFrame row with simulated values
        new_day_df = pd.DataFrame(index=[current_date_to_predict])
        new_day_df[adj_close_col_name] = simulated_next_adj_close
        new_day_df[volume_col_name] = simulated_volume
        new_day_df[open_col_name] = simulated_open
        new_day_df[high_col_name] = simulated_high
        new_day_df[low_col_name] = simulated_low
        new_day_df[close_col_name] = simulated_close

        temp_df_for_features = pd.concat([historical_window_for_forecast, new_day_df])
        temp_df_with_features = create_features(temp_df_for_features, adj_close_col_name, volume_col_name)
        temp_df_with_features.fillna(method='ffill', inplace=True)
        temp_df_with_features.fillna(method='bfill', inplace=True)

        X_current_future_day = temp_df_with_features.iloc[-1:].drop(
            columns=[col for col in features_to_drop_final if col in temp_df_with_features.columns], errors='ignore'
        )

        missing_cols = set(X_train_full.columns) - set(X_current_future_day.columns)
        for col in missing_cols:
            X_current_future_day[col] = 0
        X_current_future_day = X_current_future_day[X_train_full.columns]

        X_current_future_day.dropna(inplace=True)

        if X_current_future_day.empty:
            print(f"Warning: Features for {current_date_to_predict} became empty. Breaking forecast loop.")
            break

        X_current_future_day_scaled = scaler_X.transform(X_current_future_day)
        predicted_price = model.predict(X_current_future_day_scaled)[0]

        future_predictions.append(predicted_price)
        forecast_dates_list.append(current_date_to_predict)

        current_predicted_adj_close = predicted_price
        current_predicted_close = simulated_close

        temp_df_for_next_iteration_window = pd.DataFrame(index=[current_date_to_predict])
        temp_df_for_next_iteration_window[adj_close_col_name] = predicted_price
        temp_df_for_next_iteration_window[volume_col_name] = simulated_volume
        temp_df_for_next_iteration_window[open_col_name] = simulated_open
        temp_df_for_next_iteration_window[high_col_name] = simulated_high
        temp_df_for_next_iteration_window[low_col_name] = simulated_low
        temp_df_for_next_iteration_window[close_col_name] = simulated_close

        historical_window_for_forecast = pd.concat(
            [historical_window_for_forecast, temp_df_for_next_iteration_window]).iloc[-(max_feature_window + 1):]

    # --- Prepare output data for JSON ---
    # Filter actual data from from_date_str onwards
    from_date_dt = datetime.strptime(from_date_str, "%Y-%m-%d")
    filtered_actual_data = y[y.index >= from_date_dt]

    actual_data_list = [(date.strftime("%Y-%m-%d"), price) for date, price in filtered_actual_data.items()]
    forecast_data_list = [(date.strftime("%Y-%m-%d"), price) for date, price in
                          zip(forecast_dates_list, future_predictions)]

    forecast_stats = {
        "min": float(np.min(future_predictions)) if future_predictions else None,
        "max": float(np.max(future_predictions)) if future_predictions else None,
        "mean": float(np.mean(future_predictions)) if future_predictions else None,
        "std": float(np.std(future_predictions)) if future_predictions else None
    }

    results = {
        "ticker": symbol,
        "actual_data": actual_data_list,
        "forecast_data": forecast_data_list,
        "forecast_stats": forecast_stats,
        "last_historical_date": last_historical_date.strftime("%Y-%m-%d")
    }

    return json.dumps(results)


def plot_forecast(forecast_json_str):
    """
    Receives a JSON string containing forecast data and plots it.

    Args:
        forecast_json_str (str): JSON string returned by lightGBM_predict.
    """
    forecast_data = json.loads(forecast_json_str)

    ticker = forecast_data['ticker']
    actual_data = forecast_data['actual_data']
    forecast_data_points = forecast_data['forecast_data']
    forecast_stats = forecast_data['forecast_stats']
    last_historical_date_str = forecast_data['last_historical_date']

    actual_dates = [pd.to_datetime(dp[0]) for dp in actual_data]
    actual_prices = [dp[1] for dp in actual_data]

    forecast_dates = [pd.to_datetime(dp[0]) for dp in forecast_data_points]
    forecast_prices = [dp[1] for dp in forecast_data_points]

    last_historical_date = pd.to_datetime(last_historical_date_str)

    plt.figure(figsize=(18, 9))

    # Plot actual historical values
    plt.plot(actual_dates, actual_prices, label='Actual Historical Prices', color='blue', linewidth=2)

    # Plot future forecast
    if forecast_prices:
        plt.plot(forecast_dates, forecast_prices, label=f'Forecast (Next {len(forecast_prices)} Days)',
                 color='green', linestyle=':', linewidth=2, marker='^', markersize=4)
        plt.axvline(x=last_historical_date, color='grey', linestyle='--', linewidth=1.5, label='Forecast Start Date')

    plt.title(f"{ticker}: Actual Historical Prices and Forecast", fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.show()

    print("\nForecast Statistics:")
    for stat, value in forecast_stats.items():
        if value is not None:
            print(f"- {stat.capitalize()}: {value:.2f}")
        else:
            print(f"- {stat.capitalize()}: N/A")


# --- Example Run ---
if __name__ == "__main__":
    # Example Configuration
    SYMBOL = 'TSLA'  # Microsoft as an example
    FROM_DATE = '2024-01-01'  # Start showing actual data from this date in the plot
    DAYS_TO_PREDICT = 45  # Forecast for the next 45 days

    # Call the prediction function
    forecast_output_json = lightGBM_predict(SYMBOL, FROM_DATE, DAYS_TO_PREDICT)

    if forecast_output_json:
        # Call the plotting function with the JSON output
        plot_forecast(forecast_output_json)
    else:
        print("Prediction failed, cannot plot.")