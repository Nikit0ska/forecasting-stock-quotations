import React, { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  CartesianGrid, Legend, ResponsiveContainer, Area, ComposedChart
} from "recharts";
import {
  TextField, Button, MenuItem, Select,
  InputLabel, FormControl, Box, Typography,
  Card, CardContent, Grid, Chip, Paper,
  alpha, useTheme, IconButton, Divider
} from "@mui/material";
import {
  TrendingUp, ShowChart, Assessment,
  Timeline, BarChart, Refresh, DateRange
} from "@mui/icons-material";

const API_BASE = "http://localhost:8000";

export default function App() {
  const [assets, setAssets] = useState([]);
  const [selectedAsset, setSelectedAsset] = useState("SBER");
  const [model, setModel] = useState("naive");
  const [steps, setSteps] = useState(20);
  const [startDate, setStartDate] = useState("2025-10-01");
  const [data, setData] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const theme = useTheme();

  useEffect(() => {
    fetch(`${API_BASE}/assets`)
      .then((res) => res.json())
      .then((data) => setAssets(Object.keys(data)));
  }, []);

  const fetchData = async () => {
    if (!selectedAsset) return;
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/forecast?asset=${selectedAsset}&model=${model}&steps=${steps}&start_date=${startDate}`);
      const forecastData = await res.json();

      const chartData = [];
      const allDates = [...forecastData.dates];

      forecastData.train.forEach((value, index) => {
        chartData.push({
          date: allDates[index],
          actual: value,
          predicted: null,
        });
      });

      forecastData.test.forEach((value, index) => {
        chartData.push({
          date: allDates[forecastData.train.length + index - 1],
          actual: value,
          predicted: null,
        });
      });

      chartData.push({
        date: allDates[forecastData.train.length + forecastData.test.length - 1],
        actual: chartData[chartData.length -1].actual,
        predicted: chartData[chartData.length -1].actual,
      });

      forecastData.forecast.forEach((value, index) => {
        chartData.push({
          date: allDates[forecastData.train.length + forecastData.test.length + index],
          actual: null,
          predicted: value,
        });
      });

      setData(chartData);
      setMetrics(forecastData.metrics);
    } finally {
      setLoading(false);
    }
  };

  const gradientOffset = () => {
    if (data.length === 0) return 0;
    const dataMax = Math.max(...data.map(i => i.actual || 0));
    const dataMin = Math.min(...data.map(i => i.actual || 0));

    if (dataMax <= 0) return 0;
    if (dataMin >= 0) return 1;

    return dataMax / (dataMax - dataMin);
  };

  const off = gradientOffset();

  return (
    <Box sx={{
      minHeight: "100vh",
      background: `linear-gradient(135deg, ${alpha(theme.palette.primary.dark, 0.05)} 0%, ${alpha(theme.palette.secondary.dark, 0.02)} 100%)`,
      position: "relative",
      overflow: "hidden",
      "&::before": {
        content: '""',
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: "radial-gradient(circle at 20% 50%, rgba(33, 150, 243, 0.08) 0%, transparent 50%)",
        pointerEvents: "none"
      }
    }}>
      <Box sx={{ p: 4, position: "relative", zIndex: 1 }}>
        {/* Header */}
        <Paper
          elevation={0}
          sx={{
            mb: 4,
            p: 3,
            background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
            borderRadius: 4,
            boxShadow: `0 10px 40px ${alpha(theme.palette.primary.main, 0.3)}`,
            position: "relative",
            overflow: "hidden"
          }}
        >
          <Box sx={{ position: "relative", zIndex: 1 }}>
            <Typography
              variant="h4"
              fontWeight="bold"
              sx={{
                color: "white",
                display: "flex",
                alignItems: "center",
                gap: 1,
                mb: 1
              }}
            >
              <TrendingUp sx={{ fontSize: 35 }} />
              ИАС для моделирования и прогнозирования биржевых котировок
            </Typography>
            <Typography variant="body1" sx={{ color: alpha("#fff", 0.9), ml: 6 }}>
              Интеллектуальная система анализа и прогнозирования финансовых рынков
            </Typography>
          </Box>
          <Box
            sx={{
              position: "absolute",
              top: -20,
              right: -20,
              opacity: 0.1,
              transform: "rotate(15deg)"
            }}
          >
            <ShowChart sx={{ fontSize: 200, color: "white" }} />
          </Box>
        </Paper>

        {/* Controls */}
        <Card
          elevation={0}
          sx={{
            mb: 4,
            borderRadius: 4,
            boxShadow: `0 4px 20px ${alpha(theme.palette.common.black, 0.08)}`,
            border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
            backdropFilter: "blur(10px)",
            background: alpha(theme.palette.background.paper, 0.9)
          }}
        >
          <CardContent sx={{ p: 3 }}>
            <Grid container spacing={3} alignItems="center">
              <Grid item xs={12} md={2.5}>
                <FormControl fullWidth variant="outlined">
                  <InputLabel sx={{
                    "&.Mui-focused": { color: theme.palette.primary.main }
                  }}>
                    Актив
                  </InputLabel>
                  <Select
                    value={selectedAsset}
                    onChange={(e) => setSelectedAsset(e.target.value)}
                    label="Название актива"
                    sx={{
                      borderRadius: 3,
                      "& .MuiOutlinedInput-notchedOutline": {
                        borderColor: alpha(theme.palette.primary.main, 0.2)
                      },
                      "&:hover .MuiOutlinedInput-notchedOutline": {
                        borderColor: theme.palette.primary.main
                      }
                    }}
                  >
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
                  <InputLabel>Модель</InputLabel>
                  <Select
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    label="Модель"
                    sx={{
                      borderRadius: 3,
                      "& .MuiOutlinedInput-notchedOutline": {
                        borderColor: alpha(theme.palette.primary.main, 0.2)
                      }
                    }}
                  >
                    <MenuItem value="naive">📊 Naive</MenuItem>
                    <MenuItem value="ma">📈 Moving Average</MenuItem>
                    <MenuItem value="arima">🔮 ARIMA</MenuItem>
                    <MenuItem value="exp">📉 Exponential</MenuItem>
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} md={2}>
                <TextField
                  type="number"
                  label="Шаг прогноза"
                  value={steps}
                  onChange={(e) => setSteps(Number(e.target.value))}
                  fullWidth
                  variant="outlined"
                  sx={{
                    "& .MuiOutlinedInput-root": {
                      borderRadius: 3
                    }
                  }}
                />
              </Grid>

              <Grid item xs={12} md={2.5}>
                <TextField
                  label="Начальная дата"
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  fullWidth
                  variant="outlined"
                  InputLabelProps={{ shrink: true }}
                  sx={{
                    "& .MuiOutlinedInput-root": {
                      borderRadius: 3
                    }
                  }}
                />
              </Grid>

              <Grid item xs={12} md={2.5}>
                <Button
                  variant="contained"
                  fullWidth
                  onClick={fetchData}
                  disabled={loading}
                  startIcon={loading ? <Refresh className="spin" /> : <Timeline />}
                  sx={{
                    height: 56,
                    borderRadius: 3,
                    fontWeight: "bold",
                    background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
                    boxShadow: `0 4px 15px ${alpha(theme.palette.primary.main, 0.4)}`,
                    transition: "all 0.3s",
                    "&:hover": {
                      transform: "translateY(-2px)",
                      boxShadow: `0 6px 20px ${alpha(theme.palette.primary.main, 0.5)}`
                    },
                    "&:active": {
                      transform: "translateY(0)"
                    }
                  }}
                >
                  {loading ? "Загрузка..." : "Получить прогноз"}
                </Button>
              </Grid>
            </Grid>

            {/* Model Info Chips */}
            <Box sx={{ mt: 3, display: "flex", gap: 1, flexWrap: "wrap" }}>
              <Chip
                icon={<Assessment />}
                label={`Актив: ${selectedAsset}`}
                variant="outlined"
                sx={{ borderRadius: 2 }}
              />
              <Chip
                icon={<Timeline />}
                label={`Модель: ${model.toUpperCase()}`}
                variant="outlined"
                sx={{ borderRadius: 2 }}
              />
              <Chip
                icon={<DateRange />}
                label={`Период: ${steps} дней`}
                variant="outlined"
                sx={{ borderRadius: 2 }}
              />
            </Box>
          </CardContent>
        </Card>

        {/* Metrics */}
        {metrics && (
          <Grid container spacing={3} sx={{ mb: 4 }}>
            {[
              {
                label: "RMSE",
                value: metrics.RMSE,
                icon: "📊",
                color: "#FF6B6B",
                description: "Среднеквадратичная ошибка"
              },
              {
                label: "MAE",
                value: metrics.MAE,
                icon: "📈",
                color: "#4ECDC4",
                description: "Средняя абсолютная ошибка"
              },
              {
                label: "MAPE",
                value: metrics.MAPE + "%",
                icon: "🎯",
                color: "#45B7D1",
                description: "Средняя абсолютная процентная ошибка"
              },
            ].map((m, index) => (
              <Grid item xs={12} md={4} key={m.label}>
                <Card
                  elevation={0}
                  sx={{
                    borderRadius: 4,
                    boxShadow: `0 4px 20px ${alpha(m.color, 0.15)}`,
                    border: `1px solid ${alpha(m.color, 0.2)}`,
                    transition: "all 0.3s",
                    "&:hover": {
                      transform: "translateY(-4px)",
                      boxShadow: `0 8px 30px ${alpha(m.color, 0.25)}`
                    }
                  }}
                >
                  <CardContent sx={{ p: 3 }}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}>
                      <Box sx={{
                        width: 48,
                        height: 48,
                        borderRadius: 3,
                        background: `linear-gradient(135deg, ${alpha(m.color, 0.2)} 0%, ${alpha(m.color, 0.1)} 100%)`,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: 24
                      }}>
                        {m.icon}
                      </Box>
                      <Box>
                        <Typography variant="h6" fontWeight="bold" color="text.primary">
                          {m.label}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {m.description}
                        </Typography>
                      </Box>
                    </Box>
                    <Typography
                      variant="h3"
                      fontWeight="bold"
                      sx={{
                        color: m.color,
                        textAlign: "right"
                      }}
                    >
                      {typeof m.value === "number"
                        ? m.value.toFixed(4)
                        : m.value}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        )}

        {/* Chart */}
        <Card
          elevation={0}
          sx={{
            borderRadius: 4,
            boxShadow: `0 4px 20px ${alpha(theme.palette.common.black, 0.08)}`,
            border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
            background: alpha(theme.palette.background.paper, 0.95),
            backdropFilter: "blur(10px)"
          }}
        >
          <CardContent sx={{ p: 3 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 3 }}>
              <ShowChart sx={{ color: theme.palette.primary.main }} />
              <Typography variant="h6" fontWeight="bold">
                График прогнозирования
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
                  <linearGradient id="splitColor" x1="0" y1="0" x2="0" y2="1">
                    <stop offset={off} stopColor="#00b894" stopOpacity={0.8}/>
                    <stop offset={off} stopColor="#FF6B6B" stopOpacity={0.8}/>
                  </linearGradient>
                </defs>

                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke={alpha(theme.palette.divider, 0.5)}
                  vertical={false}
                />

                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 12, fill: theme.palette.text.secondary }}
                  axisLine={{ stroke: alpha(theme.palette.divider, 0.5) }}
                  tickLine={{ stroke: alpha(theme.palette.divider, 0.5) }}
                />

                <YAxis
                  tick={{ fontSize: 12, fill: theme.palette.text.secondary }}
                  axisLine={{ stroke: alpha(theme.palette.divider, 0.5) }}
                  tickLine={{ stroke: alpha(theme.palette.divider, 0.5) }}
                  domain={['auto', 'auto']}
                />

                <Tooltip
                  contentStyle={{
                    borderRadius: 12,
                    border: "none",
                    boxShadow: `0 8px 30px ${alpha(theme.palette.common.black, 0.12)}`,
                    background: alpha(theme.palette.background.paper, 0.95),
                    backdropFilter: "blur(10px)"
                  }}
                  labelStyle={{ fontWeight: "bold", marginBottom: 8 }}
                />

                <Legend
                  iconType="circle"
                  wrapperStyle={{ paddingTop: 20 }}
                />

                <Area
                  type="monotone"
                  dataKey="actual"
                  stroke="#3f51b5"
                  strokeWidth={2}
                  fill="url(#colorActual)"
                  dot={false}
                  name="Фактические данные"
                  activeDot={{ r: 6, fill: "#3f51b5" }}
                />

                <Line
                  type="monotone"
                  dataKey="predicted"
                  stroke="#00b894"
                  strokeWidth={3}
                  dot={false}
                  name="Прогноз"
                  strokeDasharray="5 5"
                  activeDot={{ r: 6, fill: "#00b894" }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Footer */}
        <Box sx={{ mt: 3, textAlign: "center" }}>
          <Typography variant="caption" color="text.secondary">
             • Прогноз основан на исторических данных
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