# services/prediction_service.py

import os
import requests
import yfinance as yf
from yahooquery import search as yahoo_search
import pandas as pd
from datetime import datetime, timedelta

# Import specific prediction functions
from backend.ml.predict_gru import GRU_model_predict
from backend.ml.predict_lightgbm import lightGBM_predict
from backend.ml.predict_xgboost import XGBoost_model_predict

# Allowed model keys
ALLOWED_MODELS = {
    'predict_gru',
    'predict_xgboost',
    'predict_lightgbm'
}


def search_stocks(query: str, count: int = 10) -> list[dict]:
    """
    Autocomplete search via yahooquery.search(), returns up to `count` tickers.

    :param query: Search string
    :param count: Max number of results
    :return: List of dicts: symbol, name, exchange, type
    :raises ValueError: If query is blank
    """
    q = (query or '').strip()[:50]
    if not q:
        raise ValueError("Query string is required")

    # Call yahooquery's search function
    data = yahoo_search(q) or {}
    raw_quotes = data.get('quotes', [])[:count]

    results = []
    for item in raw_quotes:
        sym = item.get('symbol')
        name = item.get('shortname') or item.get('longname')
        exch = item.get('exchDisp', '')
        typ = item.get('typeDisp', '')
        if sym and name:
            results.append({
                'symbol':   sym,
                'name':     name,
                'exchange': exch,
                'type':     typ
            })
    return results


def get_stock_quote_and_history(symbol: str, days: int = 60) -> tuple[dict, list[dict]]:
    """
    Fetch real-time quote and recent historical price data.

    :param symbol: Stock ticker symbol
    :param days: Number of past days of history to retrieve
    :return: Tuple of (quote_info, history_records)
      - quote_info: dict of ticker.info fields
      - history_records: list of daily price dicts (Open, High, Low, Close, Volume, etc.)
    :raises ValueError: If symbol is blank
    """
    sym = (symbol or '').strip()
    if not sym:
        raise ValueError("Symbol is required")

    # real-time quote
    ticker = yf.Ticker(sym)
    quote = ticker.info.copy() if hasattr(ticker, 'info') else {}
    # check existence
    if not quote or quote.get('regularMarketPrice') is None:
        raise ValueError(f"Symbol '{sym}' not found")

    # historical data
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=days)
    hist_df = ticker.history(start=start_dt, end=end_dt, interval="1d")
    if hist_df.empty:
        raise ValueError(f"No historical data for symbol '{sym}'")
    records = hist_df.reset_index().to_dict(orient="records")

    return quote, records


def run_ml_algorithm_prediction(model: str, symbol: str, future_days: int, days_from: str) -> dict:
    """
    Executes the selected ML prediction function and returns its result as a dict.

    :param model: One of 'predict_gru' or 'predict_xgboost'
    :param symbol: Stock symbol to predict for
    :param future_days: Number of days into the future to predict
    :param days_from: Date string (YYYY-MM-DD) indicating the start date for prediction
    :return: Result dict from the model
    :raises ValueError: If model is unsupported or return type is incorrect
    """
    # Delegate to specific model with error handling
    try:
        if model == 'predict_gru':
            result = GRU_model_predict(symbol, future_days, days_from)
        elif model == 'predict_xgboost':
            result = XGBoost_model_predict(symbol, future_days, days_from)
        else:  # predict_lightgbm
            result = lightGBM_predict(symbol, days_from, future_days)
    except ValueError:
        # propagate user/validation errors (e.g., symbol not found)
        raise
    except Exception as e:
        # wrap unexpected errors
        raise ValueError(f"Prediction failed for model '{model}' and symbol '{symbol}': {e}") from e

    if not isinstance(result, dict):
        raise ValueError(f"Model '{model}' must return a dict, got {type(result)}")

    return result


def get_historical_data(symbol: str, range: str = "1y", interval: str = "1d") -> list[dict]:
    """
    Fetch historical price data for a given period and interval.

    :param symbol: Stock ticker symbol
    :param range: Data range string (e.g., '1d', '5d', '1mo', '1y')
    :param interval: Interval string (e.g., '1m', '5m', '1d', '1wk')
    :return: List of price record dicts with date/time and OHLCV fields
    :raises ValueError: If symbol is blank
    """
    sym = (symbol or '').strip()
    if not sym:
        raise ValueError("Symbol is required")

    # real-time quote
    ticker = yf.Ticker(sym)
    quote = ticker.info.copy() if hasattr(ticker, 'info') else {}
    # check existence
    if not quote or quote.get('regularMarketPrice') is None:
        raise ValueError(f"Symbol '{sym}' not found")

    hist_df = ticker.history(period=range, interval=interval)
    if hist_df.empty:
        raise ValueError(f"No historical data for symbol '{sym}' with range={range} interval={interval}")

    return hist_df.reset_index().to_dict(orient="records")
