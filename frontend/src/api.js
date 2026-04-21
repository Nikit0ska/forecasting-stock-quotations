import axios from "axios";

const API_URL = "http://localhost:8000";

export async function fetchAssets() {
  const res = await axios.get(`${API_URL}/assets`);
  return res.data;
}

export async function fetchForecast(asset, model) {
  const res = await axios.get(`${API_URL}/forecast`, {
    params: { asset, model },
  });
  return res.data;
}
