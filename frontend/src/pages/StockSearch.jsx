import { useNavigate } from 'react-router-dom';
import SearchBar from '../components/SearchBar';

const StockSearch = () => {
  const navigate = useNavigate();

  const handleSelect = (symbol) => {
    navigate(`/stocks/${symbol}`);
  };

  return (
    <section className="search-page">
      <h2>Search Stocks</h2>
      <SearchBar onSearch={handleSelect} />
    </section>
  );
};

export default StockSearch;
