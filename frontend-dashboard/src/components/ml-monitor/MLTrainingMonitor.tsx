'use client';

import React, { useState } from 'react';
import { apiClient } from '../../lib/api-client';

interface Metrics {
  loss?: number;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  auprc?: number;
  roc_auc?: number;
}

export const MLTrainingMonitor: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [epochs, setEpochs] = useState(20);
  const [learningRate, setLearningRate] = useState(0.005);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [statusMessage, setStatusMessage] = useState('Siap memulai pelatihan model MHGSL.');

  const handleTrain = async () => {
    setLoading(true);
    setStatusMessage('Melatih model MHGSL pada 3 saluran graf...');
    try {
      const response = await apiClient.trainModel(epochs, 32, learningRate);
      if (response.status === 'success') {
        setMetrics(response.final_metrics);
        setHistory(response.history || []);
        setStatusMessage(`Pelatihan selesai! ${epochs} epoch dieksekusi.`);
      }
    } catch (err: any) {
      setStatusMessage(`Gagal melatih model: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 text-slate-100 shadow-xl">
      <div className="flex justify-between items-center mb-6">
        <div>
          <span className="text-xs font-semibold px-2.5 py-1 bg-emerald-950 text-emerald-400 rounded-md border border-emerald-800">
            MHGSL GCN ENGINE
          </span>
          <h2 className="text-xl font-bold mt-2">Monitor Pelatihan Deep Learning</h2>
          <p className="text-xs text-slate-400">Multi-Channel Heterogeneous Graph Structure Learning</p>
        </div>
        <button
          onClick={handleTrain}
          disabled={loading}
          className={`px-5 py-2.5 rounded-xl font-semibold text-sm transition-all shadow-md ${
            loading
              ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
              : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-900/40'
          }`}
        >
          {loading ? 'Sedang Melatih...' : 'Latih Model'}
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div>
          <label className="text-xs text-slate-400 block mb-1">Jumlah Epoch: {epochs}</label>
          <input
            type="range"
            min="5"
            max="100"
            step="5"
            value={epochs}
            onChange={(e) => setEpochs(Number(e.target.value))}
            className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer"
          />
        </div>
        <div>
          <label className="text-xs text-slate-400 block mb-1">Learning Rate: {learningRate}</label>
          <input
            type="range"
            min="0.001"
            max="0.02"
            step="0.001"
            value={learningRate}
            onChange={(e) => setLearningRate(Number(e.target.value))}
            className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer"
          />
        </div>
      </div>

      <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 text-xs font-mono text-slate-300 mb-6">
        [Status] {statusMessage}
      </div>

      {metrics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 bg-slate-950 rounded-xl border border-slate-800">
            <span className="text-[10px] text-slate-400 uppercase">Akurasi</span>
            <p className="text-2xl font-bold text-emerald-400">{(metrics.accuracy! * 100).toFixed(1)}%</p>
          </div>
          <div className="p-4 bg-slate-950 rounded-xl border border-slate-800">
            <span className="text-[10px] text-slate-400 uppercase">AUPRC</span>
            <p className="text-2xl font-bold text-blue-400">{metrics.auprc?.toFixed(4)}</p>
          </div>
          <div className="p-4 bg-slate-950 rounded-xl border border-slate-800">
            <span className="text-[10px] text-slate-400 uppercase">ROC-AUC</span>
            <p className="text-2xl font-bold text-violet-400">{metrics.roc_auc?.toFixed(4)}</p>
          </div>
          <div className="p-4 bg-slate-950 rounded-xl border border-slate-800">
            <span className="text-[10px] text-slate-400 uppercase">F1-Score</span>
            <p className="text-2xl font-bold text-amber-400">{metrics.f1_score?.toFixed(4)}</p>
          </div>
        </div>
      )}
    </div>
  );
};
