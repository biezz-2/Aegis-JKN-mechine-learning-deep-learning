"""
Data pipeline for transforming Notion data into graph-ready format for MHGSL
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .notion_client import NotionClient


class MHGSLDataPipeline:
    """Pipeline for preparing JKN claims data for MHGSL"""

    def __init__(self):
        self.notion_client = NotionClient()
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def fetch_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch data from Notion"""
        print("Fetching claims data from Notion...")
        claims = self.notion_client.fetch_claims_dataset()
        claims_df = pd.DataFrame(claims)

        print(f"Fetched {len(claims_df)} claims")

        print("Fetching research data from Notion...")
        research = self.notion_client.fetch_research_dataset()
        research_df = pd.DataFrame(research)

        print(f"Fetched {len(research_df)} research records")

        return claims_df, research_df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data"""
        # Drop rows with critical missing values
        df = df.dropna(subset=['Tanggal', 'Narasi Rekam Medis'])

        # Fill numeric missing values with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

        # Fill categorical missing values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

        return df

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for graph construction"""
        features = df.copy()

        # Convert date to numeric features
        if 'Tanggal' in features.columns:
            features['Tanggal'] = pd.to_datetime(features['Tanggal'])
            features['day_of_week'] = features['Tanggal'].dt.dayofweek
            features['month'] = features['Tanggal'].dt.month
            features['year'] = features['Tanggal'].dt.year

        # Text length features
        if 'Narasi Rekam Medis' in features.columns:
            features['narrative_length'] = features['Narasi Rekam Medis'].str.len()
            features['narrative_word_count'] = features['Narasi Rekam Medis'].str.split().str.len()

        # Sentiment score (if available)
        if 'Skor Sentimen' in features.columns:
            features['sentiment'] = features['Skor Sentimen']
        else:
            features['sentiment'] = 0.0

        return features

    def construct_graph_nodes(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Construct graph nodes for MHGSL"""
        nodes = {}

        # Patient nodes
        if 'patient_id' in df.columns:
            nodes['patients'] = df[['patient_id']].drop_duplicates().reset_index(drop=True)
        else:
            # Generate synthetic patient IDs
            df['patient_id'] = [f"P{i:04d}" for i in range(len(df))]
            nodes['patients'] = df[['patient_id']].drop_duplicates().reset_index(drop=True)

        # Doctor nodes (extract from narrative or create synthetic)
        if 'doctor_id' in df.columns:
            nodes['doctors'] = df[['doctor_id']].drop_duplicates().reset_index(drop=True)
        else:
            df['doctor_id'] = [f"D{i:04d}" for i in range(len(df) // 10)]
            nodes['doctors'] = df[['doctor_id']].drop_duplicates().reset_index(drop=True)

        # Hospital nodes
        if 'hospital_id' in df.columns:
            nodes['hospitals'] = df[['hospital_id']].drop_duplicates().reset_index(drop=True)
        else:
            df['hospital_id'] = [f"H{i:04d}" for i in range(len(df) // 5)]
            nodes['hospitals'] = df[['hospital_id']].drop_duplicates().reset_index(drop=True)

        # Procedure nodes (extract from narrative)
        if 'procedure' in df.columns:
            nodes['procedures'] = df[['procedure']].drop_duplicates().reset_index(drop=True)
        else:
            df['procedure'] = ['PROC_001'] * len(df)
            nodes['procedures'] = df[['procedure']].drop_duplicates().reset_index(drop=True)

        # Diagnosis nodes
        if 'diagnosis' in df.columns:
            nodes['diagnoses'] = df[['diagnosis']].drop_duplicates().reset_index(drop=True)
        else:
            df['diagnosis'] = ['DIAG_001'] * len(df)
            nodes['diagnoses'] = df[['diagnosis']].drop_duplicates().reset_index(drop=True)

        return nodes

    def construct_adjacency_matrices(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Construct 3-channel adjacency matrices for MHGSL"""
        # Placeholder implementation - will be expanded based on actual data structure
        num_nodes = len(df)

        # Channel 1: Topological adjacency (physical referrals)
        A_top = np.eye(num_nodes)

        # Channel 2: Feature adjacency (cosine similarity)
        A_feat = np.eye(num_nodes)

        # Channel 3: Semantic adjacency (metapath co-occurrence)
        A_sem = np.eye(num_nodes)

        return {
            'topological': A_top,
            'feature': A_feat,
            'semantic': A_sem
        }

    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete data pipeline"""
        # Fetch data
        claims_df, research_df = self.fetch_data()

        # Clean data
        claims_df = self.clean_data(claims_df)

        # Extract features
        features_df = self.extract_features(claims_df)

        # Construct graph nodes
        nodes = self.construct_graph_nodes(features_df)

        # Construct adjacency matrices
        adjacency_matrices = self.construct_adjacency_matrices(features_df)

        return {
            'features': features_df,
            'nodes': nodes,
            'adjacency_matrices': adjacency_matrices,
            'research': research_df
        }


if __name__ == "__main__":
    pipeline = MHGSLDataPipeline()
    data = pipeline.run_pipeline()

    print(f"Features shape: {data['features'].shape}")
    print(f"Nodes: {data['nodes'].keys()}")
    print(f"Adjacency matrices: {data['adjacency_matrices'].keys()}")
