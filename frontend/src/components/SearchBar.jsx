import { useState, useEffect, useRef } from 'react';
import { TextField, Button, List, ListItem, Paper, CircularProgress, Alert } from '@mui/material';
import { searchStocks, validateStock } from '../services/api';
import SearchIcon from '@mui/icons-material/Search';

const SearchBar = ({ onSearch }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [error, setError] = useState(null);
  const [validationError, setValidationError] = useState(null);
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
    setValidationError(null);

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
          setResults([]);
          setIsLoading(false);
        }
      }, 300);
    } else {
      setResults([]);
      setIsLoading(false);
    }
  };

  const handleValidateAndSearch = async (symbol) => {
    if (!symbol) return;

    setIsValidating(true);
    setValidationError(null);
    setError(null);

    try {
      const validation = await validateStock(symbol);

      if (validation.valid) {
        onSearch(symbol);
        setResults([]);
        setQuery(symbol);
      } else {
        setValidationError(validation.error || 'Stock symbol not found. Please try a different symbol.');
      }
    } catch (err) {
      console.error('Validation failed:', err);
      setValidationError('Failed to validate stock symbol. Please try again.');
    } finally {
      setIsValidating(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && query) {
      handleValidateAndSearch(query);
    }
  };

  const handleSubmit = () => {
    if (query) {
      handleValidateAndSearch(query);
    }
  };

  return (
    <div className="search-bar" style={{
      display: 'flex',
      flexDirection: 'column',
      gap: '1rem',
      position: 'relative',
      background: '#121212',
      padding: '1.2rem',
      borderRadius: '12px',
      boxShadow: '0 4px 12px rgba(100,108,255,0.5)',
      width: '100%',
    }}>
      <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
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
          disabled={isLoading || isValidating || !query}
          sx={{
            backgroundColor: '#6366F1',
            padding: '12px 24px',
            fontSize: '1.1rem',
            minWidth: '140px',
            '&:hover': {
              backgroundColor: '#4F46E5'
            }
          }}
        >
          {isLoading || isValidating ? <CircularProgress size={24} color="inherit" /> : <>
            <SearchIcon sx={{ mr: 1 }} /> Search
          </>}
        </Button>
      </div>

      {/* Error Messages */}
      {validationError && (
        <Alert severity="error" onClose={() => setValidationError(null)}>
          {validationError}
        </Alert>
      )}

      {error && (
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

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
                  handleValidateAndSearch(item.symbol);
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
    </div>
  );
};

export default SearchBar;
