"""
Runnable self-check for MHGSL, OASIS, and FastAPI.
No external test framework required: standard python assert.
"""
import asyncio
import os
import sys

# Ensure backend/src is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data.pipeline import MHGSLDataPipeline
from models.mhgsl import MHGSLModel
from oasis.simulation import JKNFraudSimulation
from fastapi.testclient import TestClient
from api.main import app

def test_pipeline():
    pipeline = MHGSLDataPipeline()
    df, _ = pipeline.fetch_data()
    assert len(df) == 300, f"Expected 300 claims, got {len(df)}"

    X, y, feats = pipeline.extract_features(df)
    assert X.shape[1] == 10, f"Expected 10 features, got {X.shape[1]}"
    assert len(y) == 300, "Target length mismatch"

    adj = pipeline.construct_adjacency_matrices(df, X)
    assert 'topological' in adj and 'feature' in adj and 'semantic' in adj, "Missing graph channels"
    assert adj['topological'].shape[0] == 2, "Edge index must have 2 rows"
    return X, adj

def test_model(X, adj):
    import torch
    model = MHGSLModel(input_dim=10, hidden_dim=32, num_classes=2, num_heads=2)
    x_tensor = torch.tensor(X, dtype=torch.float)
    logits, ch_out = model(x_tensor, adj['topological'], adj['feature'], adj['semantic'])
    assert logits.shape == (300, 2), f"Expected logits shape (300, 2), got {logits.shape}"

    contrib = model.get_channel_contributions(x_tensor, adj['topological'], adj['feature'], adj['semantic'], node_idx=0)
    assert all(k in contrib for k in ['topological', 'feature', 'semantic', 'shared']), "Missing SHAP channels"

def test_oasis():
    sim = JKNFraudSimulation()
    step_res = asyncio.run(sim.run_scenario_step("upcoding", theta_feat=0.5, shared_weight=0.7))
    assert step_res['scenario'] == 'upcoding', "Scenario mismatch"
    assert len(step_res['logs']) >= 5, "Agent step logs incomplete"
    assert 0.0 <= step_res['score'] <= 1.0, "Score out of range"

def test_api():
    client = TestClient(app)
    res_health = client.get("/health")
    assert res_health.status_code == 200, "Health check failed"
    assert res_health.json()["status"] == "healthy", "Health status invalid"

    res_sim = client.post("/api/oasis/simulate", json={"simulation_type": "upcoding", "num_agents": 10, "num_steps": 1})
    assert res_sim.status_code == 200, "Simulation API failed"

if __name__ == "__main__":
    X, adj = test_pipeline()
    test_model(X, adj)
    test_oasis()
    test_api()
    print("ALL ASSERT CHECKS PASSED: Pipeline, MHGSL, OASIS, FastAPI OK.")
