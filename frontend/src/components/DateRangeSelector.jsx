import { useState } from 'react';

const DateRangeSelector = ({ from, to, onChange }) => {
  const [fromDate, setFromDate] = useState(from || '');
  const [toDate, setToDate] = useState(to || '');

  const handleFrom = (e) => {
    const value = e.target.value;
    setFromDate(value);
    onChange({ from: value, to: toDate });
  };

  const handleTo = (e) => {
    const value = e.target.value;
    setToDate(value);
    onChange({ from: fromDate, to: value });
  };

  const datePickerStyle = {
    backgroundColor: '#1a1a1a',
    color: '#ffffff',
    border: '1px solid #333',
    borderRadius: '6px',
    padding: '12px',
    fontSize: '1.2rem',
    fontWeight: '500',
    width: '180px',
    cursor: 'pointer',
    outline: 'none',
    transition: 'all 0.3s ease',
    '&:hover': {
      borderColor: '#6366F1'
    }
  };

  const labelStyle = {
    fontSize: '1.2rem',
    fontWeight: '600',
    color: '#ffffff',
    display: 'flex',
    flexDirection: 'row',
    alignItems: 'center',
    gap: '1rem'
  };

  return (
    <div className="date-range-selector" style={{ 
      display: 'flex', 
      gap: '1.5rem', 
      alignItems: 'center', 
      justifyContent: 'center',
      backgroundColor: '#121212', 
      padding: '1.2rem', 
      borderRadius: '12px', 
      boxShadow: '0 4px 12px rgba(100,108,255,0.5)',
      width: '100%',
      maxWidth: '650px'
    }}>
      <label style={labelStyle}>
        <span>From:</span>
        <input 
          type="date" 
          value={fromDate} 
          onChange={handleFrom} 
          style={datePickerStyle}
        />
      </label>
      <label style={labelStyle}>
        <span>To:</span>
        <input 
          type="date" 
          value={toDate} 
          onChange={handleTo} 
          style={datePickerStyle}
        />
      </label>
    </div>
  );
};

export default DateRangeSelector;
