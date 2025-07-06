# services/prediction_service.py

import os
import requests
import yfinance as yf
from yahooquery import search as yahoo_search
import pandas as pd
from datetime import datetime, timedelta

# Import specific prediction functions
from backend.ml.predict_gru import GRU_model_predict
from backend.ml.predict_xgboost import XGBoost_model_predict

# Allowed model keys
ALLOWED_MODELS = {
    'predict_gru',
    'predict_xgboost',
    'predict_lightgbm'
}

# Yahoo Finance autocomplete/search endpoint
yahoo_search_url = "https://query2.finance.yahoo.com/v1/finance/search"


# def search_stocks(query: str, count: int = 10) -> list[dict]:
#     """
#     Search for stock tickers via Yahoo Finance autocomplete API.
#
#     :param query: Search string
#     :param count: Maximum number of quote results to return
#     :return: List of dicts with keys: symbol, name, exchange, type
#     :raises ValueError: If query is empty or blank
#     :raises requests.HTTPError: On bad HTTP response
#     """
#     if not query or not query.strip():
#         raise ValueError("Query string is required")
#     q = query.strip()[:50]
#
#     params = {
#         "q": q,
#         "quotesCount": count,
#         "newsCount": 0,
#         "enableNavLinks": False,
#         "enableEnhancedTrivialQuery": True,
#     }
#     resp = requests.get(yahoo_search_url, params=params, timeout=10)
#     resp.raise_for_status()
#     data = resp.json()
#     quotes = data.get("quotes", [])
#
#     results = []
#     for item in quotes:
#         sym = item.get("symbol")
#         name = item.get("shortname") or item.get("longname")
#         exch = item.get("exchDisp", "")
#         typ = item.get("typeDisp", "")
#         if sym and name:
#             results.append({
#                 "symbol": sym,
#                 "name": name,
#                 "exchange": exch,
#                 "type": typ
#             })
#     return results

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
    if not symbol or not symbol.strip():
        raise ValueError("Symbol is required")
    ticker = yf.Ticker(symbol)
    quote = ticker.info.copy()

    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=days)
    hist_df = ticker.history(start=start_dt, end=end_dt, interval="1d")
    hist_df = hist_df.reset_index()
    history = hist_df.to_dict(orient="records")

    return quote, history


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
    # Validate model
    if model not in ALLOWED_MODELS:
        raise ValueError(f"Unsupported model: {model}")

    # Delegate to the correct prediction function
    if model == 'predict_gru':
        result = GRU_model_predict(symbol, future_days, days_from)
    else:  # predict_xgboost
        result = XGBoost_model_predict(symbol, future_days, days_from)

    # Validate result type
    if not isinstance(result, dict):
        raise ValueError(
            f"Expected dict from {model} prediction function, got {type(result)}"
        )

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
    if not symbol or not symbol.strip():
        raise ValueError("Symbol is required")
    ticker = yf.Ticker(symbol)
    hist_df = ticker.history(period=range, interval=interval)
    hist_df = hist_df.reset_index()
    return hist_df.to_dict(orient="records")
