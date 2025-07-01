import { useNavigate } from 'react-router-dom';
import SearchBar from '../components/SearchBar';

const HomePage = () => {
  const navigate = useNavigate();

  const handleSelect = (symbol) => {
    navigate(`/stocks/${symbol}`);
  };

  return (
    <section className="home">
      <h1>Welcome to FinSight</h1>
      <div className="main-search-container" style={{ 
        marginTop: '2rem',
        marginBottom: '3rem',
        width: '100%',
        maxWidth: '800px',
        margin: '2rem auto'
      }}>
        <SearchBar onSearch={handleSelect} />
      </div>
      <p>
        Quickly explore real-time market data and simple predictions for your
        favourite stocks.
      </p>
    </section>
  );
};

export default HomePage;
