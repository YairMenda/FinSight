// services/predictionService.js
import { PythonShell } from 'python-shell';
import path from 'path';
import dotenv from 'dotenv';
dotenv.config();

const ALLOWED_MODELS = new Set([
  'predict_xgboost',
  'predict_gru',
  // any future scripts you add
]);

export const runMLAlgorithmPrediction = (model, symbol, futureDays, days_from) => {
  if (!ALLOWED_MODELS.has(model)) {
    return Promise.reject(new Error(`Unsupported model: ${model}`));
  }

  // Construct full path to your Python script (assuming you run `node app.js` from the `backend/` folder)
  const scriptPath = path.join(process.cwd(), 'ml', `${model}.py`);

  // Use the Python interpreter from your .venv
  const pythonPath = process.env.PYTHON_PATH || 'python';

  const args = [symbol, futureDays.toString(), days_from];

  return new Promise((resolve, reject) => {
    PythonShell.run(scriptPath, {
      pythonPath,         // ← ensures we call your venv’s python
      args,
      pythonOptions: ['-u'],  // unbuffered
      timeout: 60_000         // 60 s max
    }, (err, results) => {
      if (err) return reject(err);
      if (!results?.length) return reject(new Error('No output from Python script'));
      try {
        const raw = results[results.length - 1];
        resolve(JSON.parse(raw));
      } catch (pe) {
        reject(new Error(`Invalid JSON from Python: ${pe.message}`));
      }
    });
  });
};



// export const runGRUPrediction = (symbol, days_from, futureDays) => {
//   const scriptPath = path.join('ml', 'predict_gru.py');
//   const args = [symbol, days_from, futureDays.toString()];
//
//   return new Promise((resolve, reject) => {
//     PythonShell.run(scriptPath, { args }, (err, results) => {
//       if (err) {
//         return reject(err);
//       }
//       // handle the case where Python prints nothing
//       if (!results || results.length === 0) {
//         return reject(new Error('No output from Python script'));
//       }
//       // assume last line is our JSON
//       const raw = results[results.length - 1];
//       try {
//         const data = JSON.parse(raw);
//         return resolve(data);
//       } catch (parseErr) {
//         return reject(new Error(`Invalid JSON from Python: ${parseErr.message}`));
//       }
//     });
//   });
// };
//
// export const runXGBoostPrediction = (symbol, futureDays, days_from) => {
//   const scriptPath = path.join('ml', 'predict_xgboost.py');
//   const args = [symbol, futureDays.toString(), days_from];
//
//   return new Promise((resolve, reject) => {
//     PythonShell.run(scriptPath, { args }, (err, results) => {
//       if (err) {
//         return reject(err);
//       }
//       // handle the case where Python prints nothing
//       if (!results || results.length === 0) {
//         return reject(new Error('No output from Python script'));
//       }
//       // assume last line is our JSON
//       const raw = results[results.length - 1];
//       try {
//         const data = JSON.parse(raw);
//         return resolve(data);
//       } catch (parseErr) {
//         return reject(new Error(`Invalid JSON from Python: ${parseErr.message}`));
//       }
//     });
//   });
// };
