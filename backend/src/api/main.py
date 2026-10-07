"""
FastAPI Backend Server for Aegis-JKN ML/DL Connector
"""
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import redis
import json
import asyncio
from ..data.pipeline import MHGSLDataPipeline
from ..models.mhgsl import MHGSLModel, MHGSLTrainer
from ..oasis.simulation import JKNFraudSimulation
from ..utils.config import settings


# Initialize FastAPI app
app = FastAPI(
    title="Aegis-JKN ML/DL Connector API",
    description="Backend API for ML/DL fraud detection and OASIS simulation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis connection
redis_client = redis.from_url(settings.redis_url, decode_responses=True)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()


# Pydantic models
class TrainingRequest(BaseModel):
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001


class PredictionRequest(BaseModel):
    claim_data: Dict[str, Any]


class OASISSimulationRequest(BaseModel):
    num_agents: int = 100
    num_steps: int = 10
    simulation_type: str = "all"  # all, phantom_billing, upcoding, collusion


# Health check
@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "Aegis-JKN ML/DL Connector",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    try:
        # Check Redis connection
        redis_client.ping()
        return {
            "status": "healthy",
            "redis": "connected",
            "database": "connected"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


# Data pipeline endpoints
@app.post("/api/data/fetch")
async def fetch_data():
    """Fetch data from Notion and prepare for ML"""
    try:
        pipeline = MHGSLDataPipeline()
        data = pipeline.run_pipeline()

        # Cache in Redis
        redis_client.set("pipeline_data", json.dumps({
            "features_shape": data["features"].shape,
            "nodes_count": {k: len(v) for k, v in data["nodes"].items()}
        }, default=str))

        return {
            "status": "success",
            "features_shape": data["features"].shape,
            "nodes": list(data["nodes"].keys()),
            "adjacency_matrices": list(data["adjacency_matrices"].keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ML/DL endpoints
@app.post("/api/model/train")
async def train_model(request: TrainingRequest):
    """Train MHGSL model"""
    try:
        # Broadcast training start
        await manager.broadcast(json.dumps({
            "type": "training_started",
            "epochs": request.epochs
        }))

        # Simulate training (replace with actual training)
        for epoch in range(request.epochs):
            await asyncio.sleep(0.1)  # Simulate training time
            if epoch % 10 == 0:
                await manager.broadcast(json.dumps({
                    "type": "training_progress",
                    "epoch": epoch,
                    "total_epochs": request.epochs,
                    "loss": 0.5 - (epoch / request.epochs) * 0.4
                }))

        # Broadcast training completion
        await manager.broadcast(json.dumps({
            "type": "training_completed",
            "final_loss": 0.1,
            "accuracy": 0.91
        }))

        return {
            "status": "success",
            "epochs_trained": request.epochs,
            "final_loss": 0.1,
            "accuracy": 0.91
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/model/predict")
async def predict_fraud(request: PredictionRequest):
    """Predict fraud probability for a claim"""
    try:
        # Mock prediction (replace with actual model inference)
        fraud_probability = 0.75
        risk_level = "High" if fraud_probability > 0.7 else "Medium" if fraud_probability > 0.3 else "Low"

        return {
            "status": "success",
            "fraud_probability": fraud_probability,
            "risk_level": risk_level,
            "channel_contributions": {
                "topological": -0.12,
                "feature": 0.38,
                "semantic": 0.41,
                "shared": 0.33
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# OASIS endpoints
@app.post("/api/oasis/simulate")
async def run_oasis_simulation(request: OASISSimulationRequest):
    """Run OASIS fraud simulation"""
    try:
        simulation = JKNFraudSimulation(
            openai_api_base=settings.openai_api_base,
            openai_api_key=settings.openai_api_key,
            model_name=settings.openai_model
        )

        # Broadcast simulation start
        await manager.broadcast(json.dumps({
            "type": "simulation_started",
            "num_agents": request.num_agents,
            "num_steps": request.num_steps
        }))

        # Run simulation
        results = await simulation.run_all_simulations(
            num_agents=request.num_agents,
            num_steps=request.num_steps
        )

        # Broadcast simulation completion
        await manager.broadcast(json.dumps({
            "type": "simulation_completed",
            "results": results["summary"]
        }))

        # Compare with baseline
        comparison = simulation.compare_with_baseline(results)

        return {
            "status": "success",
            "simulation_results": results,
            "comparison_with_baseline": comparison
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back or process data
            await websocket.send_text(f"Received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Metrics endpoint
@app.get("/api/metrics")
async def get_metrics():
    """Get current metrics from Redis"""
    try:
        cached_data = redis_client.get("pipeline_data")
        if cached_data:
            return json.loads(cached_data)
        else:
            return {"status": "no_data"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers
    )
