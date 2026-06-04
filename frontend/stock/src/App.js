import React, { useEffect, useState, useMemo } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  CartesianGrid, Legend, ResponsiveContainer, Area, ComposedChart
} from "recharts";
import {
  TextField, Button, MenuItem, Select,
  InputLabel, FormControl, Box, Typography,
  Card, CardContent, Grid, Chip, Paper,
  alpha, useTheme, Divider,
  Accordion, AccordionSummary, AccordionDetails,
  Checkbox, FormControlLabel, Stack, Slider
} from "@mui/material";
import {
  TrendingUp, ShowChart, Assessment,
  Timeline, BarChart, Refresh, DateRange,
  ExpandMore, Tune, PieChart
} from "@mui/icons-material";

const API_BASE = "http://localhost:8000";

/* ====================== Гиперпараметры ====================== */
const modelParamsMeta = {
  naive: [
    { name: "seasonal_period", label: "Период сезонности", type: "number", default: 1, min: 1, step: 1,
      helperText: "Для Seasonal Naive (0 — без сезонности)" }
  ],
  ma: [
    { name: "window", label: "Размер окна", type: "number", default: 5, min: 1, step: 1 }
  ],
  exp: [
    { name: "alpha", label: "α (сглаживание уровня)", type: "number", default: 0.2, min: 0, max: 1, step: 0.05 },
    { name: "beta", label: "β (тренд)", type: "number", default: 0.1, min: 0, max: 1, step: 0.05 },
    { name: "gamma", label: "γ (сезонность)", type: "number", default: 0.1, min: 0, max: 1, step: 0.05 },
    { name: "seasonal_periods", label: "Период сезонности", type: "number", default: 7, min: 0, step: 1 },
    { name: "trend", label: "Учитывать тренд", type: "checkbox", default: false },
    { name: "seasonal", label: "Учитывать сезонность", type: "checkbox", default: false },
    { name: "model_type", label: "Тип модели", type: "select", default: "simple",
      options: ["simple", "holt", "holt-winters"] }
  ],
  arima: [
    { name: "p", label: "p (AR)", type: "number", default: 0, min: 0, step: 1 },
    { name: "d", label: "d (I)", type: "number", default: 0, min: 0, step: 1 },
    { name: "q", label: "q (MA)", type: "number", default: 0, min: 0, step: 1 },
  ],
  prophet: [
    { name: "changepoint_prior_scale", label: "changepoint_prior_scale", type: "number", default: 0.05, min: 0.001, max: 1, step: 0.01 },
    { name: "seasonality_prior_scale", label: "seasonality_prior_scale", type: "number", default: 10, min: 0.1, max: 100, step: 0.1 },
    { name: "yearly_seasonality", label: "Годовая сезонность", type: "checkbox", default: true },
    { name: "weekly_seasonality", label: "Недельная сезонность", type: "checkbox", default: true },
    { name: "daily_seasonality", label: "Дневная сезонность", type: "checkbox", default: false }
  ],
  lstm: [
    { name: "units", label: "Число нейронов", type: "number", default: 50, min: 1, step: 1 },
    { name: "epochs", label: "Эпохи", type: "number", default: 20, min: 1, step: 1 },
    { name: "batch_size", label: "Размер батча", type: "number", default: 32, min: 1, step: 1 },
    { name: "dropout", label: "Dropout", type: "number", default: 0.2, min: 0, max: 0.9, step: 0.05 },
    { name: "learning_rate", label: "Скорость обучения", type: "number", default: 0.001, min: 0.0001, max: 0.1, step: 0.0001, format: "exponential" },
    { name: "lookback", label: "Окно истории", type: "number", default: 10, min: 1, step: 1 }
  ],
  gru: [
    { name: "units", label: "Число нейронов", type: "number", default: 50, min: 1, step: 1 },
    { name: "epochs", label: "Эпохи", type: "number", default: 20, min: 1, step: 1 },
    { name: "batch_size", label: "Размер батча", type: "number", default: 32, min: 1, step: 1 },
    { name: "dropout", label: "Dropout", type: "number", default: 0.2, min: 0, max: 0.9, step: 0.05 },
    { name: "learning_rate", label: "Скорость обучения", type: "number", default: 0.001, min: 0.0001, max: 0.1, step: 0.0001, format: "exponential" },
    { name: "lookback", label: "Окно истории", type: "number", default: 10, min: 1, step: 1 }
  ],
  transformer: [
    { name: "units", label: "Число нейронов", type: "number", default: 50, min: 1, step: 1 },
    { name: "epochs", label: "Эпохи", type: "number", default: 20, min: 1, step: 1 },
    { name: "batch_size", label: "Размер батча", type: "number", default: 32, min: 1, step: 1 },
    { name: "dropout", label: "Dropout", type: "number", default: 0.2, min: 0, max: 0.9, step: 0.05 },
    { name: "learning_rate", label: "Скорость обучения", type: "number", default: 0.001, min: 0.0001, max: 0.1, step: 0.0001, format: "exponential" },
    { name: "lookback", label: "Окно истории", type: "number", default: 10, min: 1, step: 1 }
  ]
};

const getDefaultParams = (model) => {
  const meta = modelParamsMeta[model];
  if (!meta) return {};
  const defaults = {};
  meta.forEach(field => {
    defaults[field.name] = field.default;
  });
  return defaults;
};

export default function App() {
  const [assets, setAssets] = useState([]);
  const [selectedAsset, setSelectedAsset] = useState("SBER");
  const [model1, setModel1] = useState("arima");
  const [model2, setModel2] = useState("");
  const [steps, setSteps] = useState(20);
  const [startDate, setStartDate] = useState("2025-10-01");
  const [data, setData] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);

  const [hyper1, setHyper1] = useState(getDefaultParams("arima"));
  const [hyper2, setHyper2] = useState({});
  const [auto1, setAuto1] = useState(false);
  const [auto2, setAuto2] = useState(false);
  const [ensembleWeight, setEnsembleWeight] = useState(0.5);

  // Состояние для пропорций разбиения (в процентах)
  const [splitRatios, setSplitRatios] = useState({
    train: 60,
    val: 20,
    test: 20
  });

  const theme = useTheme();

  // Валидация: сумма должна быть 100%
  const splitValid = useMemo(() => {
    const sum = splitRatios.train + splitRatios.val + splitRatios.test;
    return Math.abs(sum - 100) < 0.01;
  }, [splitRatios]);

  // Обновление одного из полей разбиения
  const handleSplitChange = (field, value) => {
    setSplitRatios(prev => ({ ...prev, [field]: value }));
  };

  useEffect(() => {
    setHyper1(getDefaultParams(model1));
    setAuto1(false);
  }, [model1]);

  useEffect(() => {
    if (model2) {
      setHyper2(getDefaultParams(model2));
      setAuto2(false);
    } else {
      setHyper2({});
    }
  }, [model2]);

  useEffect(() => {
    fetch(`${API_BASE}/assets`)
      .then((res) => res.json())
      .then((data) => setAssets(Object.keys(data)));
  }, []);

  const buildSingleHyperQuery = (hyper, auto) => {
    if (auto) return "autotune=true";
    const parts = [];
    for (const [key, value] of Object.entries(hyper)) {
      parts.push(`${encodeURIComponent(key)}=${encodeURIComponent(value)}`);
    }
    return parts.join("&");
  };

  const buildEnsembleHyperQuery = (modelKey, hyper, auto) => {
    if (auto) return `${modelKey}_autotune=true`;
    const parts = [];
    for (const [key, value] of Object.entries(hyper)) {
      parts.push(`${modelKey}_${encodeURIComponent(key)}=${encodeURIComponent(value)}`);
    }
    return parts.join("&");
  };

  const fetchData = async () => {
    if (!selectedAsset || !model1 || !splitValid) return;
    setLoading(true);

    try {
      // Переводим проценты в доли
      const trainRatio = splitRatios.train / 100;
      const valRatio = splitRatios.val / 100;
      const testRatio = splitRatios.test / 100;

      const baseParams = `asset=${selectedAsset}&steps=${steps}&start_date=${startDate}&train_ratio=${trainRatio}&val_ratio=${valRatio}&test_ratio=${testRatio}`;
      let url = "";

      if (model2) {
        const hyper1Query = buildEnsembleHyperQuery("model1", hyper1, auto1);
        const hyper2Query = buildEnsembleHyperQuery("model2", hyper2, auto2);
        url = `${API_BASE}/forecast?${baseParams}&model1=${model1}&model2=${model2}&ensemble_weight=${ensembleWeight}`;
      } else {
        const hyper1Query = buildSingleHyperQuery(hyper1, auto1);
        url = `${API_BASE}/forecast?${baseParams}&model1=${model1}`;
      }

      const response = await fetch(url);
      const forecastData = await response.json();

      const chartData = [];
      const allDates = forecastData.dates;
      const trainLen = forecastData.train.length;
      const valLen = forecastData.val.length;
      const testLen = forecastData.test.length;
      const forecastLen = forecastData.forecast.length;

      for (let i = 0; i < trainLen + valLen + testLen; i++) {
        let actual = null;
        if (i < trainLen) actual = forecastData.train[i];
        else if (i < trainLen + valLen) actual = forecastData.val[i - trainLen];
        else actual = forecastData.test[i - trainLen - valLen];
        chartData.push({ date: allDates[i], actual, predicted: null });
      }

      const lastActualIdx = trainLen + valLen + testLen - 1;
      chartData.push({
        date: allDates[lastActualIdx],
        actual: chartData[lastActualIdx].actual,
        predicted: chartData[lastActualIdx].actual
      });

      for (let i = 0; i < forecastLen; i++) {
        chartData.push({
          date: allDates[lastActualIdx + 1 + i],
          actual: null,
          predicted: forecastData.forecast[i]
        });
      }

      setData(chartData);
      setMetrics(forecastData.metrics);
    } finally {
      setLoading(false);
    }
  };

  const renderHyperFields = (model, hyper, setHyper, auto, setAuto) => {
    const meta = modelParamsMeta[model];
    if (!meta) return null;

    const handleChange = (name, value) => {
      setHyper(prev => ({ ...prev, [name]: value }));
    };

    return (
      <Box>
        <FormControlLabel
          control={<Checkbox checked={auto} onChange={(e) => setAuto(e.target.checked)} />}
          label="Автоматический подбор гиперпараметров"
        />
        <Stack spacing={2} sx={{ mt: 1 }}>
          {meta.map((field) => {
            const disabled = auto;
            const value = hyper[field.name] ?? field.default;
            switch (field.type) {
              case "number":
                return (
                  <TextField
                    key={field.name}
                    label={field.label}
                    type="number"
                    value={value}
                    onChange={(e) => handleChange(field.name, parseFloat(e.target.value) || 0)}
                    disabled={disabled}
                    fullWidth
                    variant="outlined"
                    inputProps={{ min: field.min, max: field.max, step: field.step }}
                    helperText={field.helperText}
                    size="small"
                  />
                );
              case "checkbox":
                return (
                  <FormControlLabel
                    key={field.name}
                    control={
                      <Checkbox
                        checked={!!value}
                        onChange={(e) => handleChange(field.name, e.target.checked)}
                        disabled={disabled}
                      />
                    }
                    label={field.label}
                  />
                );
              case "select":
                return (
                  <FormControl fullWidth size="small" key={field.name} disabled={disabled}>
                    <InputLabel>{field.label}</InputLabel>
                    <Select
                      value={value}
                      label={field.label}
                      onChange={(e) => handleChange(field.name, e.target.value)}
                    >
                      {field.options.map(opt => (
                        <MenuItem key={opt} value={opt}>{opt}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                );
              default:
                return null;
            }
          })}
        </Stack>
      </Box>
    );
  };

  return (
    <Box sx={{
      minHeight: "100vh",
      background: `linear-gradient(135deg, ${alpha(theme.palette.primary.dark, 0.05)} 0%, ${alpha(theme.palette.secondary.dark, 0.02)} 100%)`,
      position: "relative", overflow: "hidden",
      "&::before": {
        content: '""', position: "absolute", top: 0, left: 0, right: 0, bottom: 0,
        background: "radial-gradient(circle at 20% 50%, rgba(33, 150, 243, 0.08) 0%, transparent 50%)",
        pointerEvents: "none"
      }
    }}>
      <Box sx={{ p: 4, position: "relative", zIndex: 1 }}>
        {/* Header */}
        <Paper elevation={0} sx={{ mb: 4, p: 3, background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`, borderRadius: 4, boxShadow: `0 10px 40px ${alpha(theme.palette.primary.main, 0.3)}`, position: "relative", overflow: "hidden" }}>
          <Box sx={{ position: "relative", zIndex: 1 }}>
            <Typography variant="h4" fontWeight="bold" sx={{ color: "white", display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
              <TrendingUp sx={{ fontSize: 35 }} />
              ИАС для моделирования и прогнозирования биржевых котировок
            </Typography>
            <Typography variant="body1" sx={{ color: alpha("#fff", 0.9), ml: 6 }}>
              Интеллектуальная система анализа и прогнозирования (одиночная модель или ансамбль)
            </Typography>
          </Box>
          <Box sx={{ position: "absolute", top: -20, right: -20, opacity: 0.1, transform: "rotate(15deg)" }}>
            <ShowChart sx={{ fontSize: 200, color: "white" }} />
          </Box>
        </Paper>

        {/* Controls */}
        <Card elevation={0} sx={{ mb: 4, borderRadius: 4, boxShadow: `0 4px 20px ${alpha(theme.palette.common.black, 0.08)}`, border: `1px solid ${alpha(theme.palette.divider, 0.1)}`, backdropFilter: "blur(10px)", background: alpha(theme.palette.background.paper, 0.9) }}>
          <CardContent sx={{ p: 3 }}>
            <Grid container spacing={3} alignItems="center">
              <Grid item xs={12} md={2.5}>
                <FormControl fullWidth variant="outlined">
                  <InputLabel>Актив</InputLabel>
                  <Select value={selectedAsset} onChange={(e) => setSelectedAsset(e.target.value)} label="Актив" sx={{ borderRadius: 3 }}>
                    {assets.map((asset) => (
                      <MenuItem key={asset} value={asset}>
                        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                          <BarChart sx={{ fontSize: 18, color: theme.palette.primary.main }} />
                          {asset}
                        </Box>
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={2.5}>
                <FormControl fullWidth variant="outlined">
                  <InputLabel>Модель 1</InputLabel>
                  <Select value={model1} onChange={(e) => setModel1(e.target.value)} label="Модель 1" sx={{ borderRadius: 3 }}>
                    {Object.keys(modelParamsMeta).map(m => (
                      <MenuItem key={m} value={m}>{m.toUpperCase()}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={2.5}>
                <FormControl fullWidth variant="outlined">
                  <InputLabel shrink>Модель 2 (ансамбль)</InputLabel>
                  <Select
                    value={model2}
                    onChange={(e) => setModel2(e.target.value)}
                    displayEmpty
                    renderValue={(selected) => {
                      if (!selected) {
                        return <em style={{ color: alpha(theme.palette.text.primary, 0.4) }}>Не выбрана</em>;
                      }
                      return selected.toUpperCase();
                    }}
                    sx={{ borderRadius: 3 }}
                  >
                    <MenuItem value="" disabled>
                      <em>— Не выбрана —</em>
                    </MenuItem>
                    {Object.keys(modelParamsMeta).map(m => (
                      <MenuItem key={m} value={m}>{m.toUpperCase()}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={1.5}>
                <TextField type="number" label="Шагов" value={steps} onChange={(e) => setSteps(Number(e.target.value))} fullWidth variant="outlined" sx={{ "& .MuiOutlinedInput-root": { borderRadius: 3 } }} />
              </Grid>
              <Grid item xs={12} md={2}>
                <TextField label="Начальная дата" type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} fullWidth variant="outlined" InputLabelProps={{ shrink: true }} sx={{ "& .MuiOutlinedInput-root": { borderRadius: 3 } }} />
              </Grid>
              <Grid item xs={12} md={1}>
                <Button variant="contained" fullWidth onClick={fetchData} disabled={loading || !splitValid}
                  startIcon={loading ? <Refresh className="spin" /> : <Timeline />}
                  sx={{ height: 56, borderRadius: 3, fontWeight: "bold", background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`, boxShadow: `0 4px 15px ${alpha(theme.palette.primary.main, 0.4)}`, "&:hover": { transform: "translateY(-2px)", boxShadow: `0 6px 20px ${alpha(theme.palette.primary.main, 0.5)}` }, "&:active": { transform: "translateY(0)" } }}>
                  {loading ? "Загрузка..." : "Прогноз"}
                </Button>
              </Grid>
            </Grid>

            {/* Блок разбиения данных */}
            <Box sx={{ mt: 3 }}>
              <Typography variant="subtitle1" sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
                <PieChart color="primary" fontSize="small" />
                Разбиение данных (обучающая / валидационная / тестовая)
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={4}>
                  <TextField
                    label="Обучающая, %"
                    type="number"
                    value={splitRatios.train}
                    onChange={(e) => handleSplitChange("train", Number(e.target.value))}
                    fullWidth
                    variant="outlined"
                    size="small"
                    inputProps={{ min: 0, max: 100, step: 1 }}
                    error={!splitValid}
                    helperText={!splitValid ? "Сумма должна быть 100%" : ""}
                  />
                </Grid>
                <Grid item xs={4}>
                  <TextField
                    label="Валидационная, %"
                    type="number"
                    value={splitRatios.val}
                    onChange={(e) => handleSplitChange("val", Number(e.target.value))}
                    fullWidth
                    variant="outlined"
                    size="small"
                    inputProps={{ min: 0, max: 100, step: 1 }}
                    error={!splitValid}
                  />
                </Grid>
                <Grid item xs={4}>
                  <TextField
                    label="Тестовая, %"
                    type="number"
                    value={splitRatios.test}
                    onChange={(e) => handleSplitChange("test", Number(e.target.value))}
                    fullWidth
                    variant="outlined"
                    size="small"
                    inputProps={{ min: 0, max: 100, step: 1 }}
                    error={!splitValid}
                  />
                </Grid>
              </Grid>
            </Box>

            {/* Панель гиперпараметров */}
            <Accordion sx={{ mt: 2, boxShadow: 0, border: `1px solid ${alpha(theme.palette.divider, 0.2)}`, borderRadius: 3 }}>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Tune color="primary" />
                  <Typography fontWeight="bold">
                    {model2 ? "Гиперпараметры ансамбля" : "Гиперпараметры модели"}
                  </Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={4}>
                  <Grid item xs={12} md={model2 ? 6 : 12}>
                    <Typography variant="subtitle1" gutterBottom>
                      {model1.toUpperCase()}
                    </Typography>
                    {renderHyperFields(model1, hyper1, setHyper1, auto1, setAuto1)}
                  </Grid>
                  {model2 && (
                    <Grid item xs={12} md={6}>
                      <Typography variant="subtitle1" gutterBottom>
                        {model2.toUpperCase()}
                      </Typography>
                      {renderHyperFields(model2, hyper2, setHyper2, auto2, setAuto2)}
                    </Grid>
                  )}
                </Grid>

                {model2 && (
                  <Box sx={{ mt: 3, px: 1 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Вес модели 1 в ансамбле: {ensembleWeight.toFixed(2)}
                    </Typography>
                    <Slider
                      value={ensembleWeight}
                      onChange={(e, newValue) => setEnsembleWeight(newValue)}
                      min={0}
                      max={1}
                      step={0.05}
                      marks={[
                        { value: 0, label: '0' },
                        { value: 0.5, label: '0.5' },
                        { value: 1, label: '1' }
                      ]}
                      valueLabelDisplay="auto"
                      sx={{ maxWidth: 400 }}
                    />
                    <Typography variant="caption" color="text.secondary">
                      Итоговый прогноз = вес * Прогноз₁ + (1 - вес) * Прогноз₂
                    </Typography>
                  </Box>
                )}
              </AccordionDetails>
            </Accordion>

            {/* Chips */}
            <Box sx={{ mt: 3, display: "flex", gap: 1, flexWrap: "wrap" }}>
              <Chip icon={<Assessment />} label={`Актив: ${selectedAsset}`} variant="outlined" sx={{ borderRadius: 2 }} />
              <Chip icon={<Timeline />} label={`Модель: ${model1.toUpperCase()}${model2 ? " + " + model2.toUpperCase() : ""}`} variant="outlined" sx={{ borderRadius: 2 }} />
              <Chip icon={<DateRange />} label={`Прогноз: ${steps} дн.`} variant="outlined" sx={{ borderRadius: 2 }} />
              {model2 && <Chip label={`Вес: ${ensembleWeight.toFixed(2)}`} variant="outlined" sx={{ borderRadius: 2 }} />}
            </Box>
          </CardContent>
        </Card>

        {/* Метрики */}
        {metrics && (
          <>
            <Typography variant="h6" sx={{ mb: 2, display: "flex", alignItems: "center", gap: 1 }}>
              <Assessment /> Метрики {model2 ? "ансамбля" : `модели ${model1.toUpperCase()}`}
            </Typography>
            <Grid container spacing={3} sx={{ mb: 4 }}>
              {[
                { key: "RMSE", value: metrics.RMSE, icon: "📊", color: "#FF6B6B", desc: "Среднеквадратичная ошибка" },
                { key: "MAE", value: metrics.MAE, icon: "📈", color: "#4ECDC4", desc: "Средняя абсолютная ошибка" },
                { key: "MASE", value: metrics.MASE, icon: "📉", color: "#45B7D1", desc: "Средняя абсолютная масштабированная ошибка" },
                { key: "SMAPE", value: metrics.SMAPE.toFixed(4) + "%", icon: "🎯", color: "#6C5CE7", desc: "Симметричная средняя абсолютная процентная ошибка" }
              ].map((m) => (
                <Grid item xs={12} md={3} key={m.key}>
                  <Card elevation={0} sx={{
                    borderRadius: 4, boxShadow: `0 4px 20px ${alpha(m.color, 0.15)}`,
                    border: `1px solid ${alpha(m.color, 0.2)}`,
                    transition: "all 0.3s", "&:hover": { transform: "translateY(-4px)", boxShadow: `0 8px 30px ${alpha(m.color, 0.25)}` }
                  }}>
                    <CardContent sx={{ p: 3 }}>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}>
                        <Box sx={{ width: 48, height: 48, borderRadius: 3, background: `linear-gradient(135deg, ${alpha(m.color, 0.2)} 0%, ${alpha(m.color, 0.1)} 100%)`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 24 }}>
                          {m.icon}
                        </Box>
                        <Box>
                          <Typography variant="h6" fontWeight="bold">{m.key}</Typography>
                          <Typography variant="caption" color="text.secondary">{m.desc}</Typography>
                        </Box>
                      </Box>
                      <Typography variant="h4" fontWeight="bold" sx={{ color: m.color, textAlign: "right" }}>
                        {typeof m.value === "number" ? m.value.toFixed(4) : m.value}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </>
        )}

        {/* График */}
        <Card elevation={0} sx={{ borderRadius: 4, boxShadow: `0 4px 20px ${alpha(theme.palette.common.black, 0.08)}`, border: `1px solid ${alpha(theme.palette.divider, 0.1)}`, background: alpha(theme.palette.background.paper, 0.95), backdropFilter: "blur(10px)" }}>
          <CardContent sx={{ p: 3 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 3 }}>
              <ShowChart sx={{ color: theme.palette.primary.main }} />
              <Typography variant="h6" fontWeight="bold">
                График прогноза {model2 ? "ансамбля" : ""}
              </Typography>
              <Box sx={{ flex: 1 }} />
              <Box sx={{ display: "flex", gap: 2 }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Box sx={{ width: 20, height: 3, bgcolor: "#3f51b5", borderRadius: 1 }} />
                  <Typography variant="caption">Фактические данные</Typography>
                </Box>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Box sx={{ width: 20, height: 3, bgcolor: "#00b894", borderRadius: 1 }} />
                  <Typography variant="caption">Прогноз</Typography>
                </Box>
              </Box>
            </Box>
            <Divider sx={{ mb: 3 }} />
            <ResponsiveContainer width="100%" height={450}>
              <ComposedChart data={data}>
                <defs>
                  <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3f51b5" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#3f51b5" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00b894" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#00b894" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={alpha(theme.palette.divider, 0.5)} vertical={false} />
                <XAxis dataKey="date" tick={{ fontSize: 12, fill: theme.palette.text.secondary }} />
                <YAxis tick={{ fontSize: 12, fill: theme.palette.text.secondary }} domain={['auto', 'auto']} />
                <Tooltip contentStyle={{ borderRadius: 12, border: "none", boxShadow: `0 8px 30px ${alpha(theme.palette.common.black, 0.12)}`, background: alpha(theme.palette.background.paper, 0.95), backdropFilter: "blur(10px)" }} />
                <Legend iconType="circle" wrapperStyle={{ paddingTop: 20 }} />
                <Area type="monotone" dataKey="actual" stroke="#3f51b5" strokeWidth={2} fill="url(#colorActual)" dot={false} name="Фактические данные" activeDot={{ r: 6, fill: "#3f51b5" }} />
                <Line type="monotone" dataKey="predicted" stroke="#00b894" strokeWidth={3} dot={false} name="Прогноз" strokeDasharray="5 5" activeDot={{ r: 6, fill: "#00b894" }} />
              </ComposedChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Box sx={{ mt: 3, textAlign: "center" }}>
          <Typography variant="caption" color="text.secondary">
            {model2
              ? "• Прогноз построен ансамблем двух моделей • Гиперпараметры и вес можно настроить"
              : "• Прогноз построен одной моделью • Гиперпараметры можно настроить или использовать автоподбор"}
          </Typography>
        </Box>
      </Box>
      <style jsx>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .spin {
          animation: spin 1s linear infinite;
        }
      `}</style>
    </Box>
  );
}