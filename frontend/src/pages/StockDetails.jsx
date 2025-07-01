import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getStock } from '../services/api';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import dayjs from 'dayjs';

const StockDetails = () => {
  const { symbol } = useParams();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const res = await getStock(symbol);
        const history = res.history.map((d) => ({
          date: dayjs(d.date).format('YYYY-MM-DD'),
          close: d.close,
        }));
        setData({ quote: res.quote, history });
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    })();
  }, [symbol]);

  if (loading) return <p>Loading...</p>;
  if (!data) return <p>Error loading data</p>;

  return (
    <section>
      <h2>{symbol} Details</h2>
      <p>Current Price: ${data.quote.regularMarketPrice}</p>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data.history}>
          <XAxis dataKey="date" hide />
          <YAxis domain={['auto', 'auto']} />
          <Tooltip />
          <Line type="monotone" dataKey="close" stroke="#8884d8" dot={false} />
        </LineChart>
      </ResponsiveContainer>
      <Link to={`/stocks/${symbol}/predict`}>View Prediction &rarr;</Link>
    </section>
  );
};

export default StockDetails;
