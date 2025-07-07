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
        # Prepare date ranges
        end_date = datetime.today()
        start_date = end_date - timedelta(days=3 * 365)
        from_date_dt = pd.to_datetime(from_date)

        # Download adjusted close prices
        series = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
        if series.empty:
            raise ValueError(f"No data found for symbol '{symbol}'.")

        df = pd.DataFrame(series)
        df.columns = [symbol]

        # Create rolling window features
        for window in [2, 5, 10, 20, 60]:
            df[f'{symbol}_rolling_{window}'] = df[symbol].rolling(window=window).mean().ffill().bfill()
            df[f'{symbol}_rollingSTD_{window}'] = df[symbol].rolling(window=window).std().ffill().bfill()
            df[f'{symbol}_rollingMedian_{window}'] = df[symbol].rolling(window=window).median().ffill().bfill()

        df.bfill(inplace=True)
        df.ffill(inplace=True)

        # Scale features
        X = df.drop(columns=[symbol])
        y = df[symbol]

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, shuffle=False, random_state=42
        )

        # Fit model
        model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
        model.fit(X_train, y_train)

        # Predict on test data
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        std = float(np.std(y_pred))
        r2 = r2_score(y_test, y_pred)

        # Future predictions
        last_window = X_scaled.iloc[-days_to_predict:]
        y_future_pred = model.predict(last_window)

        # Generate future dates
        last_date = y_test.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict, freq='D')

        # Prepare actual data (test set actual values)
        actual_data = []
        for date, price in y_test.items():
            if date >= from_date_dt:
                actual_data.append([date.strftime('%Y-%m-%d'), float(price)])

        # Prepare predicted data (test set predictions)
        predicted_data = []
        for date, price in zip(y_test.index, y_pred):
            if date >= from_date_dt:
                predicted_data.append([date.strftime('%Y-%m-%d'), float(price)])

        # Prepare forecasted data (future predictions)
        forecasted_data = []
        for date, price in zip(future_dates, y_future_pred):
            forecasted_data.append([date.strftime('%Y-%m-%d'), float(price)])

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
