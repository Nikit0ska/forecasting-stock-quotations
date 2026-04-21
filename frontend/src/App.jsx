import { useEffect, useState } from "react";
import { fetchAssets, fetchForecast } from "./api";
import ForecastChart from "./components/ForecastChart";
import Metrics from "./components/Metrics";
import "./App.css"; // Создайте этот файл для стилей

function App() {
  const [assets, setAssets] = useState({});
  const [asset, setAsset] = useState("SBER");
  const [model, setModel] = useState("naive");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchAssets().then(setAssets);
  }, []);

  const handleAssetClick = async (assetKey) => {
    setAsset(assetKey);
    setLoading(true);
    setError(null);

    try {
      const res = await fetchForecast(assetKey, model);
      setData(res);
    } catch (err) {
      setError("Ошибка при загрузке данных");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (Object.keys(assets).length > 0) {
      const firstAsset = Object.keys(assets)[0];
      handleAssetClick(firstAsset);
    }
  }, [assets]);

  const handleModelChange = async (e) => {
    const newModel = e.target.value;
    setModel(newModel);

    if (asset) {
      setLoading(true);
      try {
        const res = await fetchForecast(asset, newModel);
        setData(res);
      } catch (err) {
        setError("Ошибка при загрузке данных");
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <div className="app-container">
      <div className="sidebar">
        <h2>Акции</h2>
        <div className="asset-list">
          {Object.keys(assets).map(key => (
            <div
              key={key}
              className={`asset-item ${asset === key ? 'active' : ''}`}
              onClick={() => handleAssetClick(key)}
            >
              <div className="asset-symbol">{key}</div>
              <div className="asset-name">{assets[key]}</div>
            </div>
          ))}
        </div>

        <div className="model-selector">
          <h3>Модель прогноза</h3>
          <select
            value={model}
            onChange={handleModelChange}
            className="model-select"
          >
            <option value="naive">Naive</option>
            <option value="ma">Moving Average</option>
            <option value="arima">ARIMA</option>
            <option value="exp">Exponential smoothing</option>
          </select>
        </div>
      </div>

      <div className="main-content">
        <header className="header">
          <h1>Прогнозирование биржевых котировок</h1>
          <div className="current-asset">
            <span className="asset-label">Текущая акция:</span>
            <span className="asset-value">{asset} - {assets[asset]}</span>
          </div>
        </header>

        <div className="content-area">
          {loading && (
            <div className="loading">Загрузка данных...</div>
          )}

          {error && (
            <div className="error">{error}</div>
          )}

          {data && !loading && (
            <div className="forecast-section">
              <div className="chart-container">
                {data && !loading ? (
                  <ForecastChart
                    train={data.train}
                    test={data.test}
                    dates={data.dates}
                    forecast={data.forecast}
                    assetName={assets[asset]}
                  />
                ) : (
                  <div className="chart-placeholder">
                    {loading ? "Загрузка графика..." : "Выберите акцию для отображения графика"}
                  </div>
                )}
              </div>

              {data && !loading && (
                <div className="metrics-container">
                  <Metrics metrics={data.metrics} />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;