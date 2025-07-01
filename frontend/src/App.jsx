import { BrowserRouter, Routes, Route } from 'react-router-dom';

import StockSearch from './pages/StockSearch';
import StockDetails from './pages/StockDetails';
import PredictionView from './pages/PredictionView';
import Navbar from './components/Navbar';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<PredictionView />} />
        <Route path="/search" element={<StockSearch />} />
        <Route path="/stocks/:symbol" element={<StockDetails />} />
        <Route path="/stocks/:symbol/predict" element={<PredictionView />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
