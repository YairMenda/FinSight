# app.py

import os
from flask import Flask, jsonify
from flask_cors import CORS

from backend.routes.predictionController import stocks_bp  # your stocks routes as a Flask Blueprint

app = Flask(__name__)
CORS(app)  # enable Cross-Origin Resource Sharing


# Root health check
@app.route('/', methods=['GET'])
def index():
    return 'FinSight API running'


# Mount the stocks blueprint under /api/stocks
app.register_blueprint(stocks_bp, url_prefix='/api/stocks')


# Global error handler (500)
@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify({'error': 'Internal Server Error'}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(port=port)
