import { Typography, Box, Chip } from '@mui/material';

const StockDetailsPanel = ({ data, predictionData, selectedModel }) => {
  const formatNumber = (num) => {
    if (num === null || num === undefined) return 'N/A';
    return typeof num === 'number' ? num.toFixed(4) : num;
  };

  const getMetricColor = (metric, value) => {
    if (metric === 'R2') {
      return value > 0.8 ? 'success' : value > 0.5 ? 'warning' : 'error';
    }
    return 'default';
  };

  if (!data && !predictionData) return null;

  return (
    <div className="stock-details" style={{
      padding: '1.5rem',
      borderRadius: '10px',
      width: '300px',
      fontSize: '1.1rem',
      boxShadow: '0 0 8px rgba(100,108,255,0.4)',
      background: '#242424'
    }}>
      {/* Basic Stock Info */}
      {data && (
        <Box mb={2}>
          <Typography variant="h5" gutterBottom color="primary">
            {data.name || 'Stock Details'}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Min: ${formatNumber(data.min)}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Max: ${formatNumber(data.max)}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Mean: ${formatNumber(data.mean)}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Standard Deviation: {formatNumber(data.stdDeviation)}
          </Typography>
          <Typography variant="body2">
            Expected Growth:
            <span style={{
              color: data.expectedGrowth && data.expectedGrowth >= 0 ? '#4caf50' : '#f44336',
              marginLeft: '8px'
            }}>
              {data.expectedGrowth}
            </span>
          </Typography>
        </Box>
      )}

      {/* Prediction Info */}
      {predictionData && (
        <Box>
          <Typography variant="h6" gutterBottom color="secondary">
            Prediction Analysis
          </Typography>

          {selectedModel && (
            <Box mb={2}>
              <Typography variant="body2" color="text.secondary">
                Model: <strong>{selectedModel}</strong>
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Symbol: <strong>{predictionData.symbol}</strong>
              </Typography>
            </Box>
          )}

          {predictionData.metrics && (
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Model Performance:
              </Typography>

              <Box display="flex" flexWrap="wrap" gap={1} mb={1}>
                <Chip
                  label={`R²: ${formatNumber(predictionData.metrics.R2)}`}
                  color={getMetricColor('R2', predictionData.metrics.R2)}
                  size="small"
                />
              </Box>

              <Typography variant="body2" color="text.secondary">
                MAE: {formatNumber(predictionData.metrics.MAE)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                MSE: {formatNumber(predictionData.metrics.MSE)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                STD: {formatNumber(predictionData.metrics.STD)}
              </Typography>
            </Box>
          )}

          {predictionData.actual && (
            <Box mt={2}>
              <Typography variant="body2" color="text.secondary">
                Actual Data Points: {predictionData.actual.length}
              </Typography>
            </Box>
          )}

          {predictionData.predicted && (
            <Box>
              <Typography variant="body2" color="text.secondary">
                Predicted Data Points: {predictionData.predicted.length}
              </Typography>
            </Box>
          )}

          {predictionData.forecasted && (
            <Box>
              <Typography variant="body2" color="text.secondary">
                Forecasted Data Points: {predictionData.forecasted.length}
              </Typography>
            </Box>
          )}
        </Box>
      )}
    </div>
  );
};

export default StockDetailsPanel;
