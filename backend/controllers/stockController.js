// // controllers/stockController.js
// import { runXGBoostPrediction } from '../services/predictionService.js';
//
// export const getStockPrediction = async (req, res) => {
//   const { symbol, from, to } = req.query;
//
//   if (!symbol || !from || !to) {
//     return res.status(400).json({ error: 'Missing required query parameters: symbol, from, to' });
//   }
//
//   try {
//     const result = await runXGBoostPrediction(symbol, from, to, 7); // Predict 7 days ahead
//     res.json(result);
//   } catch (err) {
//     console.error('Prediction error:', err);
//     res.status(500).json({ error: 'Prediction failed: ' + err.message });
//   }
// };

import { runPythonModel } from '../services/predictionService.js';

export const getStockPrediction = async (req, res) => {
  const { symbol, from, to, model = 'xgboost', days = 7 } = req.query;

  if (!symbol || !from || !to) {
    return res.status(400).json({ error: 'Missing required query parameters: symbol, from, to' });
  }

  try {
    const result = await runPythonModel(model, symbol, from, to, parseInt(days));
    res.json(result);
  } catch (err) {
    console.error('Prediction error:', err);
    res.status(500).json({ error: 'Prediction failed: ' + err.message });
  }
};
