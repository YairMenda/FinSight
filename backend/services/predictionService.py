# services/prediction_service.py

import os

# Import specific prediction functions
from backend.ml.predict_gru import GRU_model_predict
from backend.ml.predict_xgboost import XGBoost_model_predict

# Allowed model keys
ALLOWED_MODELS = {
    'predict_gru',
    'predict_xgboost',
}


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
