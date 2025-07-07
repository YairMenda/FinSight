import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import InfoIcon from '@mui/icons-material/Info';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogActions from '@mui/material/DialogActions';
import Button from '@mui/material/Button';
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const Navbar = () => {
  const [infoOpen, setInfoOpen] = useState(false);
  const navigate = useNavigate();

  const handleInfoOpen = () => {
    setInfoOpen(true);
  };

  const handleInfoClose = () => {
    setInfoOpen(false);
  };

  const handleLogoClick = () => {
    navigate('/');
  };

  return (
    <AppBar position="fixed" color="primary" sx={{ zIndex: 1201, py: 1, boxShadow: '0 0 12px rgba(100,108,255,0.6)', background: '#242424' }}>
      <Toolbar sx={{ px: 3, display: 'flex', justifyContent: 'space-between' }}>
        <Typography
          variant="h5"
          component="div"
          sx={{
            fontWeight: 'bold',
            fontSize: '1.8rem',
            cursor: 'pointer',
            '&:hover': {
              opacity: 0.8
            }
          }}
          onClick={handleLogoClick}
        >
          FinSight
        </Typography>
        <IconButton
          color="inherit"
          onClick={handleInfoOpen}
          sx={{ ml: 2 }}
          aria-label="information"
        >
          <InfoIcon sx={{ fontSize: '1.8rem' }} />
        </IconButton>
      </Toolbar>

      <Dialog
        open={infoOpen}
        onClose={handleInfoClose}
        aria-labelledby="info-dialog-title"
        PaperProps={{
          sx: {
            bgcolor: '#1a1a1a',
            color: '#ffffff',
            borderRadius: '12px',
            boxShadow: '0 8px 32px rgba(100,108,255,0.4)',
            maxWidth: '600px'
          }
        }}
      >
        <DialogTitle id="info-dialog-title" sx={{ fontSize: '1.5rem', fontWeight: 'bold', borderBottom: '1px solid #333' }}>
          About FinSight Technologies
        </DialogTitle>
        <DialogContent sx={{ mt: 2 }}>
          <DialogContentText sx={{ color: '#e0e0e0', mb: 3 }}>
            <Typography variant="h6" sx={{ color: '#6366F1', mb: 1 }}>XGBoost</Typography>
            <Typography paragraph>
              FinSight uses XGBoost, an optimized distributed gradient boosting library, to create our predictive models. This machine learning algorithm is designed for speed and performance, allowing us to analyze complex market data and deliver accurate stock price predictions with high efficiency.
            </Typography>

            <Typography variant="h6" sx={{ color: '#6366F1', mb: 1, mt: 3 }}>Yahoo Finance Integration</Typography>
            <Typography paragraph>
              Our platform connects to Yahoo Finance APIs to retrieve real-time and historical stock data. This integration provides access to comprehensive financial information, including price history, company fundamentals, and market trends that power our analytics and visualizations.
            </Typography>

            <Typography variant="h6" sx={{ color: '#6366F1', mb: 1, mt: 3 }}>Chatbot Integration</Typography>
            <Typography paragraph>
              FinSight features an advanced natural language processing chatbot that helps users navigate financial data and understand market insights. Our AI assistant can answer questions about stocks, explain trends, and provide personalized investment recommendations based on your queries.
            </Typography>
          </DialogContentText>
        </DialogContent>
        <DialogActions sx={{ p: 2, borderTop: '1px solid #333' }}>
          <Button onClick={handleInfoClose} variant="contained" sx={{ bgcolor: '#6366F1', '&:hover': { bgcolor: '#4F46E5' } }}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </AppBar>
  );
};

export default Navbar;
