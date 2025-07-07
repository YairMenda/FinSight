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

matplotlib.use('Agg')  # Use Agg backend which doesn't require GUI
import matplotlib.pyplot as plt


def XGBoost_model_predict(symbol: str, days_to_predict: int, from_date: str) -> dict:
    try:
        from_date_dt = pd.to_datetime(from_date)
        end_date = datetime.today()
        start_date = end_date - timedelta(days=6 * 365)

        # Download data
        df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)[['Adj Close']]
        if df.empty:
            raise ValueError(f"No data found for symbol '{symbol}'.")
        df.columns = ['price']

        # Feature engineering
        for window in [2, 5, 10, 20, 60]:
            df[f'rolling_{window}'] = df['price'].rolling(window).mean()
            df[f'std_{window}'] = df['price'].rolling(window).std()
            df[f'median_{window}'] = df['price'].rolling(window).median()

        df = df.bfill().ffill()

        # Prepare features
        X = df.drop(columns=['price'])
        y = df['price']

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

        # Train on full data
        model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
        model.fit(X_scaled, y)

        # Predict full history
        y_pred_all = model.predict(X_scaled)

        # Metrics on last 10% (like GRU)
        split_index = int(0.9 * len(y))
        y_test = y[split_index:]
        y_pred_test = y_pred_all[split_index:]

        mae = mean_absolute_error(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)
        std = float(np.std(y_pred_test))
        r2 = r2_score(y_test, y_pred_test)

        # Forecasting (pct change simulation)
        pct_change_distribution = pd.Series(y_pred_all).pct_change().dropna()
        last_price = y_pred_all[-1]
        future_predictions = [last_price]
        for _ in range(days_to_predict):
            pct = np.random.choice(pct_change_distribution)
            next_price = future_predictions[-1] * (1 + pct)
            future_predictions.append(next_price)

        # Format output JSON
        actual_data = []
        predicted_data = []
        for date, true_price, pred_price in zip(df.index, y.values, y_pred_all):
            if date >= from_date_dt:
                actual_data.append([date.strftime('%Y-%m-%d'), float(true_price)])
                predicted_data.append([date.strftime('%Y-%m-%d'), float(pred_price)])

        forecast_dates = pd.date_range(df.index[-1] + timedelta(days=1), periods=days_to_predict)
        forecasted_data = [[date.strftime("%Y-%m-%d"), float(price)] for date, price in zip(forecast_dates, future_predictions[1:])]

        return {
            'symbol': symbol,
            'actual': actual_data,
            'predicted': predicted_data,
            'forecasted': forecasted_data,
            'metrics': {
                'MAE': float(mae),
                'MSE': float(mse),
                'STD': float(std),
                'R2': float(r2)
            }
        }

    except Exception as e:
        raise RuntimeError(f"Failed to predict for symbol '{symbol}': {e}") from e


def plot_predictions(result: dict):
    symbol = result['symbol']
    test_dates = pd.to_datetime(result['test_dates'])
    y_test = pd.Series(result['y_test'], index=test_dates)
    y_pred = np.array(result['y_pred'])
    future_dates = pd.to_datetime(result['future_dates'])
    y_future_pred = np.array(result['y_future_pred'])
    metrics = result['metrics']

    plt.figure(figsize=(14, 8))
    plt.plot(test_dates, y_test.values, label='Actual', marker='o')
    plt.plot(test_dates, y_pred, label='Predicted', marker='x')
    plt.plot(future_dates, y_future_pred, label=f'Forecast (next {len(y_future_pred)} days)', marker='^')
    plt.title(f"{symbol}: Actual vs Predicted vs Forecast")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    text = (
        f"R²: {metrics['r2']:.4f}\n"
        f"MAE: {metrics['mae']:.4f}\n"
        f"MSE: {metrics['mse']:.4f}\n"
        f"STD: {metrics['std']:.4f}"
    )
    plt.gcf().text(0.15, 0.75, text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    try:
        symbol = sys.argv[1]
        future_days = int(sys.argv[2])
        from_date = sys.argv[3]

        result = XGBoost_model_predict(symbol, future_days, from_date)
        sys.stdout.write(json.dumps({
            'symbol': result['symbol'],
            'plot_data': result['plot_data'],
            'metrics': result['metrics']
        }))
    except Exception as e:
        sys.stderr.write(json.dumps({'error': str(e)}))
        sys.exit(1)
