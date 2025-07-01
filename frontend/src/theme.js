import { createTheme } from '@mui/material/styles';

// Global dark theme with neon glow accents
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#646cff',
    },
    background: {
      default: '#1a1a1a',
      paper: '#242424',
    },
  },
  typography: {
    fontFamily: 'Inter, sans-serif',
  },
  shape: {
    borderRadius: 10,
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          boxShadow: '0 0 8px rgba(100, 108, 255, 0.4)',
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          boxShadow: '0 0 12px rgba(100,108,255,0.6)',
        },
      },
    },
  },
});

export default theme;
