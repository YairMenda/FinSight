import { useState, useEffect, useRef } from 'react';
import { TextField, Button, List, ListItem, Paper, CircularProgress } from '@mui/material';
import { searchStocks } from '../services/api';
import SearchIcon from '@mui/icons-material/Search';

const SearchBar = ({ onSearch }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const searchTimeout = useRef(null);
  const inputRef = useRef(null);

  // Clear results when component unmounts
  useEffect(() => {
    return () => {
      if (searchTimeout.current) {
        clearTimeout(searchTimeout.current);
      }
    };
  }, []);

  const handleChange = (e) => {
    const q = e.target.value;
    setQuery(q);
    setError(null);
    
    // Clear previous timeout
    if (searchTimeout.current) {
      clearTimeout(searchTimeout.current);
    }
    
    if (q.length >= 2) {
      setIsLoading(true);
      // Debounce search requests
      searchTimeout.current = setTimeout(async () => {
        try {
          const res = await searchStocks(q);
          setResults(res);
          setIsLoading(false);
        } catch (err) {
          console.error('Search failed:', err);
          setError('Failed to fetch search results');
          setIsLoading(false);
        }
      }, 300);
    } else {
      setResults([]);
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && query) {
      onSearch(query);
      setResults([]);
    }
  };
  
  const handleSubmit = () => {
    if (query) {
      onSearch(query);
      setResults([]);
    }
  };

  return (
    <div className="search-bar" style={{ 
      display: 'flex', 
      gap: '1rem', 
      alignItems: 'center', 
      position: 'relative', 
      background: '#121212', 
      padding: '1.2rem', 
      borderRadius: '12px', 
      boxShadow: '0 4px 12px rgba(100,108,255,0.5)',
      width: '100%',
    }}>
      <TextField
        value={query}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        placeholder="Search for stocks (e.g., AAPL, TSLA, MSFT)"
        variant="outlined"
        size="large"
        fullWidth
        inputRef={inputRef}
        InputProps={{
          style: {
            fontSize: '1.2rem',
            backgroundColor: '#1a1a1a',
            color: 'white',
            borderColor: '#333'
          }
        }}
        sx={{
          '& .MuiOutlinedInput-root': {
            '&:hover fieldset': {
              borderColor: '#6366F1',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#6366F1',
            },
          },
          '& .MuiInputLabel-root': {
            color: 'white',
          },
          '& .MuiInputBase-input': {
            color: 'white',
          },
        }}
      />
      <Button 
        variant="contained" 
        size="large" 
        onClick={handleSubmit}
        disabled={isLoading}
        sx={{ 
          backgroundColor: '#6366F1', 
          padding: '12px 24px',
          fontSize: '1.1rem',
          '&:hover': {
            backgroundColor: '#4F46E5'
          }
        }}
      >
        {isLoading ? <CircularProgress size={24} color="inherit" /> : <>
          <SearchIcon sx={{ mr: 1 }} /> Search
        </>}
      </Button>
      
      {results.length > 0 && (
        <Paper sx={{ 
          position: 'absolute', 
          top: 'calc(100% + 8px)', 
          left: 0, 
          right: 0, 
          maxHeight: 300, 
          overflow: 'auto',
          backgroundColor: '#1a1a1a',
          border: '1px solid #333',
          zIndex: 1000,
          borderRadius: '8px',
          boxShadow: '0 8px 16px rgba(0, 0, 0, 0.5)'
        }}>
          <List>
            {results.map((item) => (
              <ListItem 
                button 
                key={item.symbol} 
                onClick={() => {
                  onSearch(item.symbol);
                  setResults([]);
                  setQuery(item.symbol);
                }}
                sx={{
                  color: 'white',
                  '&:hover': {
                    backgroundColor: '#333'
                  },
                  padding: '12px 16px',
                  borderBottom: '1px solid #333',
                  '&:last-child': {
                    borderBottom: 'none'
                  }
                }}
              >
                <strong>{item.symbol}</strong> - {item.name}
              </ListItem>
            ))}
          </List>
        </Paper>
      )}
      
      {error && (
        <div style={{ position: 'absolute', top: 'calc(100% + 8px)', color: 'red', width: '100%', textAlign: 'center' }}>
          {error}
        </div>
      )}
    </div>
  );
};

export default SearchBar;
