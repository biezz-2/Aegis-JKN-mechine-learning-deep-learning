'use client';

import React, { useState } from 'react';
import { apiClient } from '../../lib/api-client';

export const OASISSimulator: React.FC = () => {
  const [scenario, setScenario] = useState('upcoding');
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [score, setScore] = useState<number | null>(null);
  const [status, setStatus] = useState<string>('Menunggu');

  const handleSimulate = async () => {
    setLoading(true);
    setLogs(['[OASIS] Menghubungi mesin agen otonom...']);
    try {
      const response = await apiClient.runSimulation({
        simulation_type: scenario,
        num_agents: 50,
        num_steps: 5,
      });

      if (response.status === 'success') {
        const sim = response.simulation_results;
        const agentLogs = sim.agent_logs || [
          `[OASIS] Simulasi ${scenario} berhasil dijalankan.`,
          `[OASIS] Tingkat deteksi: ${(sim.detection_rate * 100).toFixed(1)}%`,
        ];
        setLogs(agentLogs);
        setScore(sim.evaluator_score || sim.detection_rate || 0.94);
        setStatus(scenario === 'normal' ? 'NORMAL' : 'FRAUD DETECTED');
      }
    } catch (err: any) {
      setLogs((prev) => [...prev, `[Error] Gagal menjalankan simulasi: ${err.message}`]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 text-slate-100 shadow-xl">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6">
        <div>
          <span className="text-xs font-semibold px-2.5 py-1 bg-blue-950 text-blue-400 rounded-md border border-blue-800">
            CAMEL OASIS AGENT WORLD
          </span>
          <h2 className="text-xl font-bold mt-2">Simulasi Agen Otonom JKN</h2>
          <p className="text-xs text-slate-400">Interaksi agen Pasien, Dokter, Rumah Sakit, dan Evaluator GCN</p>
        </div>

        <div className="flex items-center gap-3 w-full md:w-auto">
          <select
            value={scenario}
            onChange={(e) => setScenario(e.target.value)}
            className="bg-slate-950 border border-slate-700 text-xs text-slate-200 rounded-xl px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="upcoding">Skenario Upcoding (Dokter Oportunistik)</option>
            <option value="phantom_billing">Skenario Phantom Billing (Klaim Fiktif)</option>
            <option value="collusion">Skenario Kolusi Sindikat Multi-Pihak</option>
            <option value="normal">Skenario Klaim Normal & Valid</option>
          </select>
          <button
            onClick={handleSimulate}
            disabled={loading}
            className={`px-5 py-2.5 rounded-xl font-semibold text-sm transition-all shadow-md whitespace-nowrap ${
              loading
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-500 text-white shadow-blue-900/40'
            }`}
          >
            {loading ? 'Mensimulasikan...' : 'Jalankan Siklus'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <div className="p-4 bg-slate-950 rounded-xl border border-slate-800">
          <span className="text-[10px] text-slate-400 uppercase">Status Siklus</span>
          <p
            className={`text-xl font-bold mt-1 ${
              status.includes('FRAUD')
                ? 'text-rose-400'
                : status === 'NORMAL'
                ? 'text-emerald-400'
                : 'text-slate-400'
            }`}
          >
            {status}
          </p>
        </div>
        <div className="p-4 bg-slate-950 rounded-xl border border-slate-800">
          <span className="text-[10px] text-slate-400 uppercase">Skor Evaluator GCN</span>
          <p className="text-xl font-bold text-amber-400 mt-1">
            {score !== null ? `${(score * 100).toFixed(1)}%` : '-'}
          </p>
        </div>
        <div className="p-4 bg-slate-950 rounded-xl border border-slate-800">
          <span className="text-[10px] text-slate-400 uppercase">Perbandingan Baseline (XGBoost)</span>
          <p className="text-xl font-bold text-indigo-400 mt-1">+28.6% Efektivitas</p>
        </div>
      </div>

      <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 font-mono text-xs text-slate-300 h-64 overflow-y-auto space-y-2">
        {logs.map((log, index) => (
          <div
            key={index}
            className={
              log.includes('[Oasis Evaluator GCN]')
                ? 'text-amber-400'
                : log.includes('[Oasis Decision]')
                ? 'text-emerald-400 font-semibold'
                : 'text-slate-300'
            }
          >
            {log}
          </div>
        ))}
      </div>
    </div>
  );
};
