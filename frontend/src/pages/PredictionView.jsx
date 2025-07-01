import { useState, useEffect } from 'react';
import SearchBar from '../components/SearchBar';
import DateRangeSelector from '../components/DateRangeSelector';
import GraphDisplay from '../components/GraphDisplay';
import StockDetailsPanel from '../components/StockDetailsPanel';
import ChatPrompt from '../components/ChatPrompt';
import { fetchMockStockData } from '../services/api';

const PredictionView = () => {
  const [symbol, setSymbol] = useState('AAPL');
  const [dates, setDates] = useState({ from: '', to: '' });
  const [stockData, setStockData] = useState(null);

  useEffect(() => {
    if (symbol) {
      fetchMockStockData(symbol).then(setStockData);
    }
  }, [symbol]);

  return (
    <div style={{ padding: '1rem', position: 'relative', marginRight: '480px' }}>
      {/* Top Controls */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginBottom: '1rem', position:'sticky', top:0, zIndex:1199 }}>
        <SearchBar onSearch={(sym) => setSymbol(sym.toUpperCase())} />
        <DateRangeSelector {...dates} onChange={setDates} />
      </div>

      {/* Main Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: '1rem' }}>
        <StockDetailsPanel data={stockData} />
        <GraphDisplay />
      </div>

      <ChatPrompt />
    </div>
  );
};

export default PredictionView;
