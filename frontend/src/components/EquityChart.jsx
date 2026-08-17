import React, { useEffect, useRef } from 'react';
import { Chart } from 'chart.js/auto';

export default function EquityChart() {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

  useEffect(() => {
    const ctx = chartRef.current.getContext('2d');

    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    chartInstance.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: ['14:00', '15:00', '16:00', '17:00', '18:00', '19:00'],
        datasets: [{
          label: 'Equity',
          data: [120000, 121400, 119800, 122300, 124850, 126200],
          borderColor: '#00f5ff',
          borderWidth: 3,
          fill: true,
          backgroundColor: 'rgba(0, 245, 255, 0.08)',
          tension: 0.4,
          pointRadius: 0,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: '#27251f' }, ticks: { color: '#52525b' } },
          y: { 
            grid: { color: '#27251f' }, 
            ticks: { color: '#52525b', callback: v => '$' + (v/1000) + 'k' } 
          }
        }
      }
    });

    return () => {
      if (chartInstance.current) chartInstance.current.destroy();
    };
  }, []);

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-3xl p-6">
      <div className="flex justify-between items-center mb-4">
        <div className="font-semibold">Equity Curve (24h)</div>
        <div className="text-xs px-3 py-1 bg-zinc-800 rounded-2xl text-zinc-400">LIVE</div>
      </div>
      <div className="h-64">
        <canvas ref={chartRef}></canvas>
      </div>
    </div>
  );
}