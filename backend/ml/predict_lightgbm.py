import sys
import yfinance as yf
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def LightGBM_model_predict(symbol: str, days_to_predict: int, from_date: str) -> dict:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=6 * 365)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    from_date_dt = pd.to_datetime(from_date)

    series = yf.download(symbol, start=start_str, end=end_str, auto_adjust=False)['Adj Close']
    df = pd.DataFrame(series)
    df.columns = [symbol]

    for window in [2, 5, 10, 20, 60]:
        df[f'{symbol}_rolling_{window}'] = df[symbol].rolling(window=window).mean().ffill().bfill()
        df[f'{symbol}_rollingSTD_{window}'] = df[symbol].rolling(window=window).std().ffill().bfill()
        df[f'{symbol}_rollingMedian_{window}'] = df[symbol].rolling(window=window).median().ffill().bfill()

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    X = df.drop(columns=[symbol])
    y = df[symbol]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        index=X.index,
        columns=X.columns
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False, random_state=42
    )

    model = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    last_window = X_scaled.iloc[-days_to_predict:]
    y_future_pred = model.predict(last_window)

    test_dates = y_test.index
    future_start = test_dates[-1] + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=future_start, periods=days_to_predict, freq='D')

    plot_data = []
    for date, actual, pred in zip(test_dates, y_test, y_pred):
        if date >= from_date_dt:
            plot_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'actual': float(actual),
                'predicted': float(pred)
            })
    for date, forecast in zip(future_dates, y_future_pred):
        if date >= from_date_dt:
            plot_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'forecast': float(forecast)
            })

    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'std': float(np.std(y_pred))
    }

    test_dates_list = [d.strftime('%Y-%m-%d') for d in test_dates]
    y_test_list = [float(x) for x in y_test]
    y_pred_list = [float(x) for x in y_pred]
    future_dates_list = [d.strftime('%Y-%m-%d') for d in future_dates]
    y_future_list = [float(x) for x in y_future_pred]

    return {
        'symbol': symbol,
        'plot_data': plot_data,
        'metrics': metrics,
        'test_dates': test_dates_list,
        'y_test': y_test_list,
        'y_pred': y_pred_list,
        'future_dates': future_dates_list,
        'y_future_pred': y_future_list
    }


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

        result = LightGBM_model_predict(symbol, future_days, from_date)

        print(json.dumps({
            'symbol': result['symbol'],
            'plot_data': result['plot_data'],
            'metrics': result['metrics']
        }))

        plot_predictions(result)

    except Exception as e:
        print(json.dumps({'error': str(e)}))
        sys.exit(1)
