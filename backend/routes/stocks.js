import express from 'express';
import yahooFinance from 'yahoo-finance2';
import dayjs from 'dayjs';
import { rateLimit } from 'express-rate-limit';

const router = express.Router();

// Rate limiting middleware to prevent API abuse
const apiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests, please try again later.' }
});

// GET /api/stocks/search?q=... -> basic symbol search
router.get('/search', apiLimiter, async (req, res) => {
  const query = req.query.q;
  
  // Input validation
  if (!query || query.trim() === '') {
    return res.status(400).json({ error: 'Query param q is required' });
  }
  
  // Sanitize input - ensure only necessary characters are used
  const sanitizedQuery = query.trim().substring(0, 50); // Limit query length
  
  try {
    // Make the Yahoo Finance API call
    const results = await yahooFinance.search(sanitizedQuery, { 
      count: 10, // Limit results
      newsCount: 0, // Don't need news
      quotesCount: 10, // Just focus on quotes
      enableNavLinks: false,
      enableEnhancedTrivialQuery: true
    });
    
    // Process and transform the results
    if (!results || !results.quotes || !Array.isArray(results.quotes)) {
      return res.json([]);
    }
    
    const simplified = results.quotes
      .filter(quote => quote && quote.symbol && quote.shortname) // Filter out any invalid entries
      .map(({ symbol, shortname, exchDisp, typeDisp }) => ({ 
        symbol, 
        name: shortname,
        exchange: exchDisp || '',
        type: typeDisp || ''
      }));
    
    res.json(simplified);
  } catch (err) {
    console.error('Yahoo Finance search error:', err);
    res.status(500).json({ error: 'Failed to fetch search results' });
  }
});

// GET /api/stocks/:symbol -> real-time quote + recent historical data
router.get('/:symbol', async (req, res) => {
  const { symbol } = req.params;
  try {
    // quote
    const quote = await yahooFinance.quote(symbol);
    // last 60 days daily prices
    const to = new Date();
    const from = dayjs(to).subtract(60, 'day').toDate();
    const history = await yahooFinance.historical(symbol, { period1: from, period2: to });
    res.json({ quote, history });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to fetch stock data' });
  }
});

// naive prediction: simple moving average forecast
router.get('/:symbol/predict', async (req, res) => {
  const { symbol } = req.params;
  try {
    const to = new Date();
    const from = dayjs(to).subtract(30, 'day').toDate();
    const history = await yahooFinance.historical(symbol, { period1: from, period2: to });
    const closePrices = history.map((d) => d.close).filter(Boolean);
    const avg = closePrices.reduce((a, b) => a + b, 0) / closePrices.length;
    // Return avg as next day prediction
    res.json({ symbol, prediction: { date: dayjs(to).add(1, 'day').format('YYYY-MM-DD'), price: avg } });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to generate prediction' });
  }
});

export default router;
