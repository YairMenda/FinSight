import { PythonShell } from 'python-shell';
import path from 'path';

export const runGRUPrediction = (symbol, days_from, futureDays) => {
  const scriptPath = path.join('ml', 'gru.py');
  const args = [symbol, days_from, futureDays.toString()];

  return new Promise((resolve, reject) => {
    PythonShell.run(scriptPath, { args }, (err, results) => {
      if (err) {
        return reject(err);
      }
      // handle the case where Python prints nothing
      if (!results || results.length === 0) {
        return reject(new Error('No output from Python script'));
      }
      // assume last line is our JSON
      const raw = results[results.length - 1];
      try {
        const data = JSON.parse(raw);
        return resolve(data);
      } catch (parseErr) {
        return reject(new Error(`Invalid JSON from Python: ${parseErr.message}`));
      }
    });
  });
};
