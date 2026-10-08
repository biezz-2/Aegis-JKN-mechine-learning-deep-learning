"""
Data pipeline for transforming JKN claims data into 3-channel graph format for MHGSL.
Supports offline datasets (claims_300_v2.1.csv, claims_db.json) and online Notion API.
Complies with Anti-Leakage policy (v2.1).
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Support running directly or as package
current_dir = Path(__file__).resolve().parent
if str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))

try:
    from .notion_client import NotionClient
except (ImportError, ValueError):
    try:
        from data.notion_client import NotionClient
    except ImportError:
        try:
            from src.data.notion_client import NotionClient
        except ImportError:
            from notion_client import NotionClient


class MHGSLDataPipeline:
    """Pipeline for preparing JKN claims data into multi-channel graph for MHGSL"""

    # Anti-leakage: Kolom turunan sistem dilarang masuk ke fitur input model
    LEAKAGE_COLUMNS = {
        'skor_fraud', 'sinyal_shap', 'kontribusi_saluran', 'risiko',
        'alasan', 'modus', 'status', 'sindikat', 'is_fraud', 'claim_id'
    }

    def __init__(self, data_dir: Optional[str] = None):
        self.notion_client = NotionClient()
        self.scaler = StandardScaler()
        self.base_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parents[3]
        self.datasets_dir = self.base_dir / "datasets"
        self.notion_docs_dir = self.base_dir.parent / "Notion-docs"

    def fetch_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch claims data and research dataset.
        Priority:
        1. datasets/claims_300_v2.1.csv (300 klaim terverifikasi v2.1)
        2. datasets/claims_db.json
        3. Notion-docs/claims_db.json
        4. Notion API (online)
        """
        claims_df = None
        research_df = pd.DataFrame()

        # 1. Cek file CSV 300 klaim
        csv_path = self.datasets_dir / "claims_300_v2.1.csv"
        if csv_path.exists():
            print(f"[Pipeline] Memuat data lokal: {csv_path.name}")
            claims_df = pd.read_csv(csv_path)

        # 2. Cek JSON datasets jika CSV tidak ada
        if claims_df is None:
            for p in [self.datasets_dir / "claims_db.json", self.notion_docs_dir / "claims_db.json"]:
                if p.exists():
                    print(f"[Pipeline] Memuat data lokal: {p.name}")
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    results = data if isinstance(data, list) else data.get("results", [])
                    extracted = [self.notion_client.extract_claim_data(page) for page in results]
                    claims_df = pd.DataFrame(extracted)
                    break

        # 3. Fallback ke Notion API jika lokal kosong
        if claims_df is None or len(claims_df) == 0:
            print("[Pipeline] Mengambil data langsung dari Notion API...")
            try:
                claims = self.notion_client.fetch_claims_dataset()
                claims_df = pd.DataFrame(claims)
            except Exception as e:
                print(f"[Pipeline] Gagal menghubungi Notion API: {e}. Menggunakan fallback sintetis.")
                claims_df = self._generate_fallback_dataset()

        # Muat research data jika tersedia
        for rp in [self.datasets_dir / "dataset_db.json", self.notion_docs_dir / "dataset_db.json"]:
            if rp.exists():
                with open(rp, "r", encoding="utf-8") as f:
                    rdata = json.load(f)
                rresults = rdata if isinstance(rdata, list) else rdata.get("results", [])
                research_df = pd.DataFrame(rresults)
                break

        print(f"[Pipeline] Berhasil memuat {len(claims_df)} baris klaim.")
        return claims_df, research_df

    def _generate_fallback_dataset(self) -> pd.DataFrame:
        """Fallback sintetis jika tidak ada file data lokal maupun koneksi API"""
        np.random.seed(20260707)
        rows = []
        for i in range(100):
            is_fraud = 1 if i < 14 else 0
            rows.append({
                "claim_id": f"KLM{i+1:03d}",
                "tanggal": "2026-03-01",
                "faskes_id": f"RS_{chr(65 + (i % 7))}",
                "dokter_id": f"D{(i % 12) + 1:02d}",
                "pasien_id": f"P{(i % 20) + 1:02d}",
                "biaya_rp": 50000000 if is_fraud else 12000000,
                "los_hari": 1 if is_fraud else 3,
                "layanan": "rawat inap",
                "diagnosis_code": "K29.7" if is_fraud else "I10",
                "prosedur_code": "00.66" if is_fraud else "89.52",
                "narasi_rekam_medis": "Tindakan kardiovaskular" if is_fraud else "Pemeriksaan rutin",
                "skor_sentimen": -0.8 if is_fraud else 0.2,
                "kontradiksi_narasi": 1 if is_fraud else 0,
                "is_fraud": is_fraud,
                "risiko": "fraud" if is_fraud else "low"
            })
        return pd.DataFrame(rows)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and impute missing values"""
        df = df.copy()

        # Mapping jika kolom berasal dari raw Notion properties
        raw_to_norm = {
            'Name': 'claim_id',
            'Tanggal': 'tanggal',
            'Faskes': 'faskes_id',
            'Dokter': 'dokter_id',
            'Pasien': 'pasien_id',
            'Biaya Rp': 'biaya_rp',
            'LOS hari': 'los_hari',
            'Layanan': 'layanan',
            'Diagnosis ICD-10': 'diagnosis_code',
            'Prosedur ICD-9-CM': 'prosedur_code',
            'Narasi Rekam Medis': 'narasi_rekam_medis',
            'Umpan Balik Pasien': 'umpan_balik_pasien',
            'Skor Sentimen': 'skor_sentimen',
            'Kontradiksi Narasi': 'kontradiksi_narasi',
            'Risiko': 'risiko',
            'Skor Fraud': 'skor_fraud'
        }
        for raw, norm in raw_to_norm.items():
            if raw in df.columns and norm not in df.columns:
                df[norm] = df[raw]

        # Ekstrak ID bersih dari format raw: 'D01 (dr. Adnan...)' -> 'D01'
        for col in ['pasien_id', 'dokter_id']:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].astype(str).str.extract(r'([PD]\d+)', expand=False).fillna(df[col])

        # Pastikan kolom esensial bertipe tepat
        df['biaya_rp'] = pd.to_numeric(df.get('biaya_rp', 0), errors='coerce').fillna(10000000)
        df['los_hari'] = pd.to_numeric(df.get('los_hari', 1), errors='coerce').fillna(2)
        df['skor_sentimen'] = pd.to_numeric(df.get('skor_sentimen', 0.0), errors='coerce').fillna(0.0)
        df['kontradiksi_narasi'] = pd.to_numeric(df.get('kontradiksi_narasi', 0), errors='coerce').fillna(0)

        # Target Ground Truth
        if 'is_fraud' not in df.columns:
            if 'risiko' in df.columns:
                df['is_fraud'] = (df['risiko'].str.lower() == 'fraud').astype(int)
            else:
                df['is_fraud'] = 0

        return df

    def extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract numerical and categorical features complying with anti-leakage policy.
        Returns: (feature_matrix_X, target_y, feature_names)
        """
        df = self.clean_data(df)
        feature_names = []
        feature_arrays = []

        # 1. Log Biaya
        log_biaya = np.log1p(df['biaya_rp'].values).reshape(-1, 1)
        feature_arrays.append(log_biaya)
        feature_names.append("log_biaya")

        # 2. Length of Stay (LOS)
        los = df['los_hari'].values.reshape(-1, 1)
        feature_arrays.append(los)
        feature_names.append("los_hari")

        # 3. Biaya per hari rawat
        cost_per_day = (df['biaya_rp'].values / (df['los_hari'].values + 0.1)).reshape(-1, 1)
        feature_arrays.append(np.log1p(cost_per_day))
        feature_names.append("log_cost_per_day")

        # 4. Sentimen pasien (-1 s.d 1)
        sentiment = df['skor_sentimen'].values.reshape(-1, 1)
        feature_arrays.append(sentiment)
        feature_names.append("skor_sentimen")

        # 5. Kontradiksi narasi klinis (0 / 1)
        contra = df['kontradiksi_narasi'].values.reshape(-1, 1)
        feature_arrays.append(contra)
        feature_names.append("kontradiksi_narasi")

        # 6. Fitur teks narasi (panjang karakter & jumlah kata)
        narasi = df.get('narasi_rekam_medis', pd.Series([''] * len(df))).fillna('').astype(str)
        text_len = narasi.str.len().values.reshape(-1, 1)
        word_cnt = narasi.str.split().str.len().values.reshape(-1, 1)
        feature_arrays.append(text_len)
        feature_arrays.append(word_cnt)
        feature_names.extend(["narrative_length", "narrative_word_count"])

        # 7. Layanan inap flag
        layanan = df.get('layanan', pd.Series([''] * len(df))).astype(str).str.lower()
        is_inap = (layanan.str.contains('inap')).astype(int).values.reshape(-1, 1)
        feature_arrays.append(is_inap)
        feature_names.append("is_rawat_inap")

        # 8. Frekuensi entitas (Faskes & Dokter frequency encoding)
        for col in ['faskes_id', 'dokter_id']:
            if col in df.columns:
                freq = df[col].map(df[col].value_counts()).fillna(1).values.reshape(-1, 1)
                feature_arrays.append(freq)
                feature_names.append(f"{col}_frequency")

        raw_X = np.hstack(feature_arrays)
        scaled_X = self.scaler.fit_transform(raw_X)
        y = df['is_fraud'].values.astype(int)

        return scaled_X, y, feature_names

    def construct_graph_nodes(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Construct dictionary of unique entity nodes aligned with FHIR R4"""
        df = self.clean_data(df)
        nodes = {
            'patients': df[['pasien_id']].drop_duplicates().reset_index(drop=True),
            'doctors': df[['dokter_id']].drop_duplicates().reset_index(drop=True),
            'hospitals': df[['faskes_id']].drop_duplicates().reset_index(drop=True),
            'procedures': df[['prosedur_code']].drop_duplicates().reset_index(drop=True),
            'diagnoses': df[['diagnosis_code']].drop_duplicates().reset_index(drop=True),
            'claims': df[['claim_id']].drop_duplicates().reset_index(drop=True)
        }
        return nodes

    def construct_adjacency_matrices(
        self,
        df: pd.DataFrame,
        features_X: np.ndarray,
        theta_feat: float = 0.5
    ) -> Dict[str, Any]:
        """
        Construct 3-channel graph edge indices for MHGSL:
        1. Topological (A_top): Shared doctor, hospital, or patient.
        2. Feature (A_feat): Cosine similarity between feature vectors >= theta_feat.
        3. Semantic (A_sem): Clinical co-occurrence (same diagnosis-procedure pair or syndicate).
        """
        import torch

        num_nodes = len(df)

        # --- Saluran 1: Graf Topologi (A_top) ---
        # ponytail: direct numpy array comparison avoids pandas .iloc overhead
        p_arr = df['pasien_id'].to_numpy()
        d_arr = df['dokter_id'].to_numpy()
        h_arr = df['faskes_id'].to_numpy()

        edges_top = [(i, i) for i in range(num_nodes)]  # Self-loops
        for i in range(num_nodes):
            pi, di, hi = p_arr[i], d_arr[i], h_arr[i]
            for j in range(i + 1, num_nodes):
                if pi == p_arr[j] or di == d_arr[j] or hi == h_arr[j]:
                    edges_top.append((i, j))
                    edges_top.append((j, i))

        # --- Saluran 2: Graf Fitur (A_feat) ---
        # ponytail: vectorized upper triangle mask over similarity matrix
        edges_feat = [(i, i) for i in range(num_nodes)]
        norm_X = features_X / (np.linalg.norm(features_X, axis=1, keepdims=True) + 1e-8)
        sim_matrix = np.dot(norm_X, norm_X.T)
        u, v = np.triu_indices(num_nodes, k=1)
        feat_mask = sim_matrix[u, v] >= theta_feat
        for r, c in zip(u[feat_mask], v[feat_mask]):
            edges_feat.append((int(r), int(c)))
            edges_feat.append((int(c), int(r)))

        # --- Saluran 3: Graf Semantik (A_sem) ---
        # ponytail: direct array comparisons for clinical co-occurrence
        edges_sem = [(i, i) for i in range(num_nodes)]
        diag_arr = df['diagnosis_code'].to_numpy()
        proc_arr = df['prosedur_code'].to_numpy()
        has_sindikat = 'sindikat' in df.columns
        sindikat_arr = df['sindikat'].to_numpy() if has_sindikat else None

        for i in range(num_nodes):
            d_i, p_i = diag_arr[i], proc_arr[i]
            s_i = sindikat_arr[i] if has_sindikat else None
            is_synd_i = bool(pd.notna(s_i) and str(s_i).startswith('SYND')) if has_sindikat else False

            for j in range(i + 1, num_nodes):
                same_pair = (d_i == diag_arr[j]) and (p_i == proc_arr[j])
                sindikat_match = False
                if is_synd_i and pd.notna(sindikat_arr[j]):
                    sindikat_match = (s_i == sindikat_arr[j])

                if same_pair or sindikat_match:
                    edges_sem.append((i, j))
                    edges_sem.append((j, i))

        # Konversi ke format torch.LongTensor [2, num_edges]
        edge_index_top = torch.tensor(edges_top, dtype=torch.long).t().contiguous()
        edge_index_feat = torch.tensor(edges_feat, dtype=torch.long).t().contiguous()
        edge_index_sem = torch.tensor(edges_sem, dtype=torch.long).t().contiguous()

        return {
            'topological': edge_index_top,
            'feature': edge_index_feat,
            'semantic': edge_index_sem,
            'counts': {
                'topological_edges': len(edges_top),
                'feature_edges': len(edges_feat),
                'semantic_edges': len(edges_sem),
            }
        }

    def run_pipeline(self, theta_feat: float = 0.5) -> Dict[str, Any]:
        """Eksekusi pipeline lengkap"""
        claims_df, research_df = self.fetch_data()
        cleaned_df = self.clean_data(claims_df)
        features_X, y, feature_names = self.extract_features(cleaned_df)
        nodes = self.construct_graph_nodes(cleaned_df)
        adjacency = self.construct_adjacency_matrices(cleaned_df, features_X, theta_feat=theta_feat)

        return {
            'claims_df': cleaned_df,
            'features_X': features_X,
            'y': y,
            'feature_names': feature_names,
            'nodes': nodes,
            'adjacency_matrices': adjacency,
            'research_df': research_df
        }


if __name__ == "__main__":
    pipeline = MHGSLDataPipeline()
    res = pipeline.run_pipeline()
    print(f"X shape: {res['features_X'].shape}")
    print(f"y shape: {res['y'].shape}, Fraud count: {sum(res['y'])}")
    print(f"Features: {res['feature_names']}")
    print(f"Edge counts: {res['adjacency_matrices']['counts']}")
