export default function Metrics({ metrics }) {
  if (!metrics) return null;

  return (
    <div>
      <h3>Метрики качества</h3>
      <ul>
        <li>RMSE: {metrics.RMSE.toFixed(3)}</li>
        <li>MAE: {metrics.MAE.toFixed(3)}</li>
        <li>MAPE: {metrics.MAPE.toFixed(2)}%</li>
      </ul>
    </div>
  );
}
