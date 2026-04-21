import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  Title,
  CategoryScale,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  LineElement,
  PointElement,
  LinearScale,
  Title,
  CategoryScale,
  Tooltip,
  Legend
);

export default function ForecastChart({ train, test, forecast, assetName, dates }) {

  const options = {
    responsive: true,
    animations: {
    x: { duration: 0 },
    y: { duration: 0 },
    active: { duration: 0 }, // 👈 Отключаем анимацию при наведении
    },
    transitions: {
        active: {
          animation: {
            duration: 0 // 👈 Без анимации при смене активных элементов
          }
        }
    },
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          usePointStyle: true,
        },
      },
      title: {
        display: true,
        text: `Прогноз котировок: ${assetName || 'акция'}`,
        font: {
          size: 18,
          weight: 'bold',
        },
      },
      tooltip: {
        callbacks: {
          label: (context) =>
            `${context.dataset.label}: ${context.parsed.y?.toFixed(2)}`,
        },
      },
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(0,0,0,0.05)',
        },
        title: {
          display: true,
          text: 'Даты',
        },
      },
      y: {
        grid: {
          color: 'rgba(0,0,0,0.05)',
        },
        title: {
          display: true,
          text: 'Цена',
        },
      },
    },
  };

      const labels = [
      ...Array.from({ length: train.length + test.length + forecast.length - 3 }, (_, i) => dates.at(i)),

    ];

    const data = {
      labels,
      datasets: [
        {
          label: "Historical",
          data: train,
          borderColor: '#1f77b4',
          backgroundColor: 'rgba(31,119,180,0.1)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.2,
        },
        {
          label: "Validation",
          data: [...Array(train.length-1).fill(null), ...test],
          borderColor: '#2ca02c',
          backgroundColor: 'rgba(44,160,44,0.1)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.2,
        },
        {
          label: "Forecast",
          data: [
            ...Array(train.length-1).fill(null),
            ...forecast,
          ],
          borderColor: '#d62728',
          borderDash: [6, 6],
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.2,
        },
      ],
    };

    console.log("test");
  return <Line data={data} options={options} />;
}
