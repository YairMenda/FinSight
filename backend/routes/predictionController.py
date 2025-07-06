# routes/predictionController.py

import logging

import requests
from flask import Blueprint, request, jsonify

from backend.services.predictionService import (
    search_stocks,
    get_stock_quote_and_history,
    run_ml_algorithm_prediction,
    get_historical_data)

logger = logging.getLogger(__name__)
stocks_bp = Blueprint('stocks', __name__)


@stocks_bp.route('/search', methods=['GET'])
def search():
    q = request.args.get('q', '', type=str).strip()
    if not q:
        return jsonify({'error': 'Query param q is required'}), 400

    sanitized = q[:50]

    try:
        results = search_stocks(sanitized, count=10)
        return jsonify(results)

    except requests.exceptions.HTTPError as he:
        status = he.response.status_code if he.response is not None else None
        if status == 429:
            logger.warning('Yahoo Finance rate limit reached: %s', he)
            return jsonify({'error': 'Too many requests, please try again later.'}), 429
        logger.error('Yahoo Finance HTTP error: %s', he, exc_info=True)
        return jsonify({'error': 'Failed to fetch search results'}), 500

    except Exception as e:
        logger.error('Yahoo Finance search error: %s', e, exc_info=True)
        return jsonify({'error': 'Failed to fetch search results'}), 500


@stocks_bp.route('/<string:symbol>', methods=['GET'])
def quote_and_history(symbol):
    try:
        quote, history = get_stock_quote_and_history(symbol, days=60)
        return jsonify({'quote': quote, 'history': history})
    except Exception as e:
        logger.error('Error fetching data for %s: %s', symbol, e, exc_info=True)
        return jsonify({'error': 'Failed to fetch stock data'}), 500


@stocks_bp.route('/<string:algorithm>/<string:symbol>/predict/<int:future_days>/<string:days_from>', methods=['GET'])
def predict(algorithm, symbol, future_days, days_from):
    # days_from expected in YYYY-MM-DD format
    try:
        result = run_ml_algorithm_prediction(
            algorithm,
            symbol,
            future_days,
            days_from
        )
        return jsonify(result)
    except ValueError as ve:
        # e.g., invalid date or params
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error('Prediction error [%s]: %s', algorithm, e, exc_info=True)
        return jsonify({
            'error': f'Failed to run {algorithm} prediction: {e}'
        }), 500


@stocks_bp.route('/<string:symbol>/history', methods=['GET'])
def history(symbol):
    range_param = request.args.get('range', '1y')
    interval = request.args.get('interval', '1d')

    try:
        data = get_historical_data(symbol, range_param, interval)
        return jsonify(data)
    except Exception as e:
        logger.error('Error fetching historical data for %s: %s', symbol, e, exc_info=True)
        return jsonify({'error': 'Failed to fetch historical data'}), 500
