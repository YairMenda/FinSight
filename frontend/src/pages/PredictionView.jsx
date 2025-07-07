import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Button,
  Box,
  Alert,
  CircularProgress
} from '@mui/material';
import SearchBar from '../components/SearchBar';
import DateRangeSelector from '../components/DateRangeSelector';
import GraphDisplay from '../components/GraphDisplay';
import StockDetailsPanel from '../components/StockDetailsPanel';
import ChatPrompt from '../components/ChatPrompt';
import { predictStock, PREDICTION_MODELS, getStock } from '../services/api';

const PredictionView = () => {
  const { symbol: urlSymbol } = useParams();
  const [symbol, setSymbol] = useState(urlSymbol || 'AAPL');
  const [selectedModel, setSelectedModel] = useState('predict_xgboost');
  const [futureDays, setFutureDays] = useState(30);
  const [fromDate, setFromDate] = useState('2024-01-01');
  const [stockData, setStockData] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Helper function to calculate min/max from prediction data arrays
  const calculateMinMaxFromPredictionData = (data) => {
    if (!data) return { min: 0, max: 0, mean: 0 };

    const { actual = [], predicted = [], forecasted = [] } = data;
    const allPrices = [];

    // Extract all price values from the arrays
    actual.forEach(item => allPrices.push(item[1]));
    predicted.forEach(item => allPrices.push(item[1]));
    forecasted.forEach(item => allPrices.push(item[1]));

    if (allPrices.length === 0) return { min: 0, max: 0, mean: 0 };

    const min = Math.min(...allPrices);
    const max = Math.max(...allPrices);
    const mean = allPrices.reduce((sum, price) => sum + price, 0) / allPrices.length;

    return { min, max, mean };
  };

  // Helper function to calculate standard deviation from prediction data
  const calculateStandardDeviation = (data) => {
    if (!data) return 0;

    const { actual = [], predicted = [], forecasted = [] } = data;
    const allPrices = [];

    actual.forEach(item => allPrices.push(item[1]));
    predicted.forEach(item => allPrices.push(item[1]));
    forecasted.forEach(item => allPrices.push(item[1]));

    if (allPrices.length === 0) return 0;

    const mean = allPrices.reduce((sum, price) => sum + price, 0) / allPrices.length;
    const variance = allPrices.reduce((sum, price) => sum + Math.pow(price - mean, 2), 0) / allPrices.length;

    return Math.sqrt(variance);
  };

  const handlePredict = async () => {
    if (!symbol || !selectedModel) return;

    setLoading(true);
    setError(null);

    try {
      const result = await predictStock(selectedModel, symbol, futureDays, fromDate);
      setPredictionData(result);
    } catch (err) {
      console.error('Prediction failed:', err);
      setError(err.response?.data?.error || 'Failed to fetch prediction data');
    } finally {
      setLoading(false);
    }
  };

  // Fetch stock data when symbol changes
  useEffect(() => {
    if (symbol) {
      setError(null);

      getStock(symbol)
        .then(data => {
          // Transform the data to match what StockDetailsPanel expects
          const transformedData = {
            name: symbol.toUpperCase(),
            min: data.quote.fiftyTwoWeekLow || 0,
            max: data.quote.fiftyTwoWeekHigh || 0,
            mean: data.quote.regularMarketPrice || 0,
            stdDeviation: 0, // We could calculate this from history if needed
            expectedGrowth: data.quote.trailingPE ? `${data.quote.trailingPE.toFixed(2)} P/E` : 'N/A'
          };
          setStockData(transformedData);
        })
        .catch(err => {
          console.error('Failed to fetch stock data:', err);
          setError('Failed to fetch stock data');
        });
    }
  }, [symbol]);

  // Update stock data when prediction data changes
  useEffect(() => {
    if (predictionData && stockData) {
      const { min, max, mean } = calculateMinMaxFromPredictionData(predictionData);
      const stdDeviation = calculateStandardDeviation(predictionData);

      setStockData(prev => ({
        ...prev,
        min: min,
        max: max,
        mean: mean,
        stdDeviation: stdDeviation
      }));
    }
  }, [predictionData]);

  useEffect(() => {
    if (urlSymbol) {
      setSymbol(urlSymbol);
    }
  }, [urlSymbol]);

  // Auto-trigger prediction when symbol is provided via URL
  useEffect(() => {
    if (urlSymbol && !predictionData && !loading) {
      // Small delay to ensure the symbol state is updated
      const timer = setTimeout(() => {
        handlePredict();
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [urlSymbol, predictionData, loading]);

  const handleSymbolChange = (newSymbol) => {
    setSymbol(newSymbol.toUpperCase());
    setPredictionData(null); // Clear previous prediction data
    setError(null);
  };

  return (
    <div style={{ padding: '1rem', position: 'relative', marginRight: '480px' }}>
      {/* Top Controls */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem',
        marginBottom: '1rem',
        position: 'sticky',
        top: 0,
        zIndex: 1199,
        backgroundColor: '#121212',
        padding: '1rem',
        borderRadius: '8px'
      }}>
        <SearchBar onSearch={handleSymbolChange} />

        <Box display="flex" gap={2} flexWrap="wrap" alignItems="center">
          <FormControl variant="outlined" size="small" sx={{ minWidth: 200 }}>
            <InputLabel>Prediction Model</InputLabel>
            <Select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              label="Prediction Model"
            >
              {Object.entries(PREDICTION_MODELS).map(([key, label]) => (
                <MenuItem key={key} value={key}>
                  {label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <TextField
            label="Days to Predict"
            type="number"
            value={futureDays}
            onChange={(e) => setFutureDays(parseInt(e.target.value) || 30)}
            variant="outlined"
            size="small"
            sx={{ minWidth: 150 }}
            inputProps={{ min: 1, max: 365 }}
          />

          <TextField
            label="From Date"
            type="date"
            value={fromDate}
            onChange={(e) => setFromDate(e.target.value)}
            variant="outlined"
            size="small"
            InputLabelProps={{ shrink: true }}
            sx={{ minWidth: 150 }}
          />

          <Button
            variant="contained"
            onClick={handlePredict}
            disabled={loading || !symbol}
            size="large"
            sx={{ minWidth: 120 }}
          >
            {loading ? <CircularProgress size={24} /> : 'Predict'}
          </Button>
        </Box>

        {error && (
          <Alert severity="error" onClose={() => setError(null)}>
            {error}
          </Alert>
        )}
      </div>

      {/* Main Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: '1rem' }}>
        <StockDetailsPanel
          data={stockData}
          predictionData={predictionData}
          selectedModel={selectedModel ? PREDICTION_MODELS[selectedModel] : null}
        />
        <GraphDisplay
          predictionData={predictionData}
          loading={loading}
        />
      </div>

      <ChatPrompt />
    </div>
  );
};

export default PredictionView;
