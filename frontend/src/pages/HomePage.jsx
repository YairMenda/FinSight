import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Button,
  Grid,
  Paper,
  Container,
  Chip
} from '@mui/material';
import SearchBar from '../components/SearchBar';
import { PREDICTION_MODELS } from '../services/api';

const HomePage = () => {
  const navigate = useNavigate();
  const [selectedSymbol, setSelectedSymbol] = useState('');

  const handleSelect = (symbol) => {
    setSelectedSymbol(symbol);
    navigate(`/predict/${symbol}`);
  };

  const handleQuickPredict = (symbol) => {
    navigate(`/predict/${symbol}`);
  };

  const popularStocks = [
    { symbol: 'AAPL', name: 'Apple Inc.' },
    { symbol: 'TSLA', name: 'Tesla Inc.' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.' },
    { symbol: 'MSFT', name: 'Microsoft Corp.' },
    { symbol: 'AMZN', name: 'Amazon.com Inc.' },
    { symbol: 'NVDA', name: 'NVIDIA Corp.' }
  ];

  return (
    <Container maxWidth="lg">
      <Box py={4} pt={12}>
        {/* Hero Section */}
        <Box textAlign="center" mb={6}>
          <Typography variant="h2" component="h1" gutterBottom color="primary">
            Welcome to FinSight
          </Typography>
          <Typography variant="h5" color="text.secondary" paragraph>
            AI-Powered Stock Analysis & Prediction Platform
          </Typography>
          <Typography variant="body1" color="text.secondary" paragraph>
            Explore real-time market data and generate sophisticated predictions
            using advanced machine learning models
          </Typography>
        </Box>

        {/* Search Section */}
        <Grid container spacing={4}>
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 4, background: '#1a1a1a' }}>
              <Typography variant="h4" gutterBottom color="secondary" textAlign="center">
                Start Your Analysis
              </Typography>
              <Box maxWidth="600px" mx="auto">
                <SearchBar onSearch={handleSelect} />
              </Box>
            </Paper>
          </Grid>

          {/* Popular Stocks */}
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 4, background: '#1a1a1a' }}>
              <Typography variant="h5" gutterBottom color="secondary">
                Popular Stocks
              </Typography>
              <Grid container spacing={2}>
                {popularStocks.map((stock) => (
                  <Grid item xs={12} sm={6} md={4} key={stock.symbol}>
                    <Paper
                      elevation={2}
                      sx={{
                        p: 2,
                        background: '#2a2a2a',
                        cursor: 'pointer',
                        '&:hover': {
                          background: '#3a3a3a'
                        }
                      }}
                      onClick={() => handleSelect(stock.symbol)}
                    >
                      <Typography variant="h6" color="primary">
                        {stock.symbol}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {stock.name}
                      </Typography>
                      <Box mt={2} display="flex" gap={1}>
                        <Button
                          size="small"
                          variant="contained"
                          onClick={(e) => {
                            e.stopPropagation();
                            navigate(`/predict/${stock.symbol}`);
                          }}
                        >
                          View
                        </Button>
                      </Box>
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            </Paper>
          </Grid>

          {/* Features Section */}
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 4, background: '#1a1a1a' }}>
              <Typography variant="h5" gutterBottom color="secondary">
                Advanced ML Models
              </Typography>
              <Grid container spacing={3}>
                {Object.entries(PREDICTION_MODELS).map(([key, label]) => (
                  <Grid item xs={12} md={4} key={key}>
                    <Box
                      p={3}
                      sx={{
                        background: '#2a2a2a',
                        borderRadius: 2,
                        height: '100%'
                      }}
                    >
                      <Typography variant="h6" color="primary" gutterBottom>
                        {label}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {key === 'predict_gru' && 'Deep learning model with memory cells, excellent for sequential time series data'}
                        {key === 'predict_xgboost' && 'Gradient boosting framework optimized for structured data with engineered features'}
                        {key === 'predict_lightgbm' && 'Fast gradient boosting framework with high efficiency and accuracy'}
                      </Typography>
                    </Box>
                  </Grid>
                ))}
              </Grid>
            </Paper>
          </Grid>

          {/* Quick Access */}
          <Grid item xs={12}>
            <Box display="flex" justifyContent="center" gap={3}>
              <Button
                variant="contained"
                color="primary"
                size="large"
                onClick={() => navigate('/predict')}
              >
                Start Prediction
              </Button>
            </Box>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
};

export default HomePage;
