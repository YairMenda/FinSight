import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || 'http://localhost:5000/api',
});

export const searchStocks = (q) => api.get('/stocks/search', { params: { q } }).then(r => r.data);
export const getStock = (symbol) => api.get(`/stocks/${symbol}`).then(r => r.data);
export const predictStock = (symbol) => api.get(`/stocks/${symbol}/predict`).then(r => r.data);
export const getHistory = (symbol, range = '1y', interval = '1d') => api.get(`/stocks/${symbol}/history`, { params: { range, interval }}).then(r => r.data);

// --- Mock helpers for UI prototype ---
const mockStockData = {
  name: 'AAPL',
  min: 120.34,
  max: 179.22,
  mean: 150.67,
  stdDeviation: 15.21,
  expectedGrowth: '5.4% yearly',
};

export const fetchMockStockData = (symbol) => {
  return Promise.resolve({ ...mockStockData, name: symbol.toUpperCase() });
};

export default api;
