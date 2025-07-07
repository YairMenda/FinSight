import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || 'http://localhost:5000/api',
});

export const searchStocks = (q) => api.get('/stocks/search', { params: { q } }).then(r => r.data);
export const getStock = (symbol) => api.get(`/stocks/${symbol}`).then(r => r.data);

// Updated prediction API to match backend endpoint
export const predictStock = (algorithm, symbol, futureDays = 30, daysFrom = '2024-01-01') => {
  return api.get(`/stocks/${algorithm}/${symbol}/predict/${futureDays}/${daysFrom}`).then(r => r.data);
};

export const getHistory = (symbol, range = '1y', interval = '1d') => api.get(`/stocks/${symbol}/history`, { params: { range, interval }}).then(r => r.data);

// Available prediction models
export const PREDICTION_MODELS = {
  'predict_gru': 'GRU Neural Network',
  'predict_xgboost': 'XGBoost',
  'predict_lightgbm': 'LightGBM'
};

export default api;
