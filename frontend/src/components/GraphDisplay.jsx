import { useState } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { Button, ButtonGroup, Typography, Box } from '@mui/material';

const GraphDisplay = ({ predictionData, loading }) => {
  const [visibleSeries, setVisibleSeries] = useState({
    actual: true,
    predicted: true,
    forecasted: true
  });

  // Convert backend data format to chart format
  const formatChartData = (data) => {
    if (!data) return [];

    const { actual = [], predicted = [], forecasted = [] } = data;
    const chartData = [];

    // Create a map of all dates
    const dateMap = new Map();

    // Add actual data
    actual.forEach(([date, price]) => {
      dateMap.set(date, { date, actual: price });
    });

    // Add predicted data
    predicted.forEach(([date, price]) => {
      if (dateMap.has(date)) {
        dateMap.get(date).predicted = price;
      } else {
        dateMap.set(date, { date, predicted: price });
      }
    });

    // Add forecasted data
    forecasted.forEach(([date, price]) => {
      if (dateMap.has(date)) {
        dateMap.get(date).forecasted = price;
      } else {
        dateMap.set(date, { date, forecasted: price });
      }
    });

    // Convert to array and sort by date
    return Array.from(dateMap.values()).sort((a, b) => new Date(a.date) - new Date(b.date));
  };

  const toggleSeries = (series) => {
    setVisibleSeries(prev => ({
      ...prev,
      [series]: !prev[series]
    }));
  };

  const showAll = () => {
    setVisibleSeries({ actual: true, predicted: true, forecasted: true });
  };

  const hideAll = () => {
    setVisibleSeries({ actual: false, predicted: false, forecasted: false });
  };

  const chartData = formatChartData(predictionData);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height={400}>
        <Typography>Loading prediction data...</Typography>
      </Box>
    );
  }

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <Box mb={2}>
        <Typography variant="h6" gutterBottom>
          Stock Price Prediction
        </Typography>
        <ButtonGroup variant="outlined" size="small" sx={{ mb: 2 }}>
          <Button
            onClick={showAll}
            variant="contained"
            size="small"
          >
            Show All
          </Button>
          <Button
            onClick={hideAll}
            variant="outlined"
            size="small"
          >
            Hide All
          </Button>
        </ButtonGroup>
        <ButtonGroup variant="outlined" size="small" sx={{ ml: 2 }}>
          <Button
            onClick={() => toggleSeries('actual')}
            variant={visibleSeries.actual ? 'contained' : 'outlined'}
            color="primary"
          >
            Actual
          </Button>
          <Button
            onClick={() => toggleSeries('predicted')}
            variant={visibleSeries.predicted ? 'contained' : 'outlined'}
            color="secondary"
          >
            Predicted
          </Button>
          <Button
            onClick={() => toggleSeries('forecasted')}
            variant={visibleSeries.forecasted ? 'contained' : 'outlined'}
            color="success"
          >
            Forecasted
          </Button>
        </ButtonGroup>
      </Box>

      <div style={{ width: '100%', height: 400 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 20, right: 30, left: 60, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="date"
              label={{ value: 'Date', position: 'insideBottom', offset: -10 }}
              tick={{ fontSize: 12 }}
            />
            <YAxis
              domain={['dataMin - 5', 'dataMax + 5']}
              tick={{ fontSize: 12 }}
              label={{ value: 'Stock Price ($)', angle: -90, position: 'insideLeft' }}
              tickFormatter={(value) => `$${value.toFixed(2)}`}
            />
            <Tooltip
              labelFormatter={(value) => `Date: ${value}`}
              formatter={(value, name) => [`$${value?.toFixed(2)}`, name]}
            />
            <Legend />

            {visibleSeries.actual && (
              <Line
                type="monotone"
                dataKey="actual"
                stroke="#1976d2"
                strokeWidth={2}
                dot={false}
                name="Actual Price"
              />
            )}

            {visibleSeries.predicted && (
              <Line
                type="monotone"
                dataKey="predicted"
                stroke="#9c27b0"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={false}
                name="Predicted Price"
              />
            )}

            {visibleSeries.forecasted && (
              <Line
                type="monotone"
                dataKey="forecasted"
                stroke="#2e7d32"
                strokeWidth={2}
                strokeDasharray="10 10"
                dot={false}
                name="Forecasted Price"
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default GraphDisplay;
