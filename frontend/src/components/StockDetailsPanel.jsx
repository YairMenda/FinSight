const StockDetailsPanel = ({ data }) => {
  if (!data) return null;
  const { name, min, max, mean, stdDeviation, expectedGrowth } = data;

  return (
    <div className="stock-details" style={{ padding: '1.5rem', borderRadius: '10px', width: '300px', fontSize: '1.1rem', boxShadow: '0 0 8px rgba(100,108,255,0.4)', background:'#242424' }}>
      <h3>{name}</h3>
      <p>Min: {min}</p>
      <p>Max: {max}</p>
      <p>Mean: {mean}</p>
      <p>Standard Deviation: {stdDeviation}</p>
      <p>Expected Growth: <span style={{ color: expectedGrowth >= 0 ? '#4caf50' : '#f44336' }}>{expectedGrowth}</span></p>
    </div>
  );
};

export default StockDetailsPanel;
