import { BrowserRouter, Routes, Route } from 'react-router-dom';

import HomePage from './pages/HomePage';
import PredictionView from './pages/PredictionView';
import Navbar from './components/Navbar';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/predict" element={<PredictionView />} />
        <Route path="/predict/:symbol" element={<PredictionView />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
