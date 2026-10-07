# API Reference

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is implemented. In production, implement JWT or API key authentication.

## Endpoints

### Health Check

#### GET /health

Check service health and database connections.

**Response:**
```json
{
  "status": "healthy",
  "redis": "connected",
  "database": "connected"
}
```

### Data Pipeline

#### POST /api/data/fetch

Fetch data from Notion and prepare for ML pipeline.

**Request Body:**
```json
{}
```

**Response:**
```json
{
  "status": "success",
  "features_shape": [100, 16],
  "nodes": ["patients", "doctors", "hospitals", "procedures", "diagnoses"],
  "adjacency_matrices": ["topological", "feature", "semantic"]
}
```

### Model Training

#### POST /api/model/train

Train MHGSL model.

**Request Body:**
```json
{
  "epochs": 100,
  "batch_size": 32,
  "learning_rate": 0.001
}
```

**Response:**
```json
{
  "status": "success",
  "epochs_trained": 100,
  "final_loss": 0.1,
  "accuracy": 0.91
}
```

**WebSocket Events:**
During training, WebSocket events are broadcast:
```json
{
  "type": "training_started",
  "epochs": 100
}
```
```json
{
  "type": "training_progress",
  "epoch": 10,
  "total_epochs": 100,
  "loss": 0.45
}
```
```json
{
  "type": "training_completed",
  "final_loss": 0.1,
  "accuracy": 0.91
}
```

### Fraud Prediction

#### POST /api/model/predict

Predict fraud probability for a claim.

**Request Body:**
```json
{
  "claim_data": {
    "patient_id": "P001",
    "doctor_id": "D001",
    "procedure": "PROC_001",
    "diagnosis": "DIAG_001",
    "amount": 1000000,
    "narrative": "Patient received treatment..."
  }
}
```

**Response:**
```json
{
  "status": "success",
  "fraud_probability": 0.75,
  "risk_level": "High",
  "channel_contributions": {
    "topological": -0.12,
    "feature": 0.38,
    "semantic": 0.41,
    "shared": 0.33
  }
}
```

### OASIS Simulation

#### POST /api/oasis/simulate

Run OASIS fraud simulation.

**Request Body:**
```json
{
  "num_agents": 100,
  "num_steps": 10,
  "simulation_type": "all"
}
```

**simulation_type options:**
- `all` - Run all simulations
- `phantom_billing` - Phantom billing pattern
- `upcoding` - Upcoding pattern
- `collusion` - Multi-party collusion pattern

**Response:**
```json
{
  "status": "success",
  "simulation_results": {
    "model_used": true,
    "simulations": {
      "phantom_billing": {...},
      "upcoding": {...},
      "collusion": {...}
    },
    "summary": {
      "total_fraud_claims": 150,
      "average_detection_rate": 0.88
    }
  },
  "comparison_with_baseline": {
    "baseline_auprc": 0.71,
    "oasis_detection_rate": 0.88,
    "improvement_percentage": 23.94,
    "oasis_improves_detection": true
  }
}
```

**WebSocket Events:**
```json
{
  "type": "simulation_started",
  "num_agents": 100,
  "num_steps": 10
}
```
```json
{
  "type": "simulation_completed",
  "results": {
    "total_fraud_claims": 150,
    "average_detection_rate": 0.88
  }
}
```

### Metrics

#### GET /api/metrics

Get current metrics from Redis cache.

**Response:**
```json
{
  "features_shape": [100, 16],
  "nodes_count": {
    "patients": 50,
    "doctors": 10,
    "hospitals": 5,
    "procedures": 20,
    "diagnoses": 15
  }
}
```

### WebSocket

#### WS /ws

WebSocket endpoint for real-time monitoring.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
};
```

**Event Types:**
- `training_started` - Training started
- `training_progress` - Training progress update
- `training_completed` - Training completed
- `simulation_started` - Simulation started
- `simulation_completed` - Simulation completed

## Error Responses

All endpoints may return error responses:

```json
{
  "detail": "Error message here"
}
```

**HTTP Status Codes:**
- `200` - Success
- `400` - Bad Request
- `500` - Internal Server Error
- `503` - Service Unavailable
