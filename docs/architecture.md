# Architecture Documentation

## System Architecture

Aegis-JKN ML/DL Connector menggunakan arsitektur microservices dengan komponen berikut:

## Components

### 1. Data Pipeline Module
- **Purpose**: Ekstraksi dan preprocessing data dari Notion API
- **Input**: Notion database (Dataset Riset, Klaim Aegis-JKN)
- **Output**: Graph-ready features dan adjacency matrices
- **Tech Stack**: Python, Pandas, Notion Client

### 2. ML/DL Model Module
- **Purpose**: Implementasi MHGSL untuk fraud detection
- **Architecture**:
  - Channel-specific GCNs (3 channels: Topological, Feature, Semantic)
  - Shared-parameter GCN
  - Multi-channel attention fusion
  - Classification head dengan SHAP explainability
- **Tech Stack**: PyTorch, PyTorch Geometric, SHAP

### 3. OASIS Integration Module
- **Purpose**: Simulasi agent-based untuk fraud pattern analysis
- **Components**:
  - Agent Mapper: Map JKN entities → OASIS agents
  - Simulation Engine: Run fraud pattern simulations
  - Analyzer: Compare simulation results with baseline
- **Tech Stack**: camel-oasis, OpenAI API, Asyncio

### 4. API Server Module
- **Purpose**: RESTful API dan WebSocket untuk frontend integration
- **Endpoints**:
  - `/api/data/fetch` - Data pipeline
  - `/api/model/train` - Model training
  - `/api/model/predict` - Fraud prediction
  - `/api/oasis/simulate` - OASIS simulation
  - `/ws` - WebSocket for real-time updates
- **Tech Stack**: FastAPI, Uvicorn, WebSockets

### 5. Database Layer
- **Redis**: Caching dan real-time metrics
- **PostgreSQL**: Persistent storage untuk training results dan simulation logs

## Data Flow

```
Notion API → Data Pipeline → Graph Construction → MHGSL Model → Prediction
                                                                    ↓
WebSocket ← API Server ← OASIS Simulation ← Agent Mapping ← Notion API
```

## Deployment Architecture

```
┌─────────────────────────────────────────┐
│     Local Development Environment       │
│                                         │
│  ┌──────────────┐    ┌──────────────┐  │
│  │ FastAPI      │    │ PostgreSQL   │  │
│  │ Backend      │◄──►│ Database     │  │
│  │ :8000        │    │ :5432        │  │
│  └──────┬───────┘    └──────────────┘  │
│         │                                │
│         │ WebSocket                     │
│         ↓                                │
│  ┌──────────────┐    ┌──────────────┐  │
│  │ Redis        │    │ Aegis-JKN    │  │
│  │ :6379        │    │ Vercel App   │  │
│  └──────────────┘    └──────────────┘  │
└─────────────────────────────────────────┘
```

## Scalability Considerations

1. **Horizontal Scaling**: FastAPI workers dapat di-scale menggunakan multiple processes
2. **Database Pooling**: Connection pooling untuk PostgreSQL
3. **Redis Clustering**: Untuk high-throughput scenarios
4. **Async Processing**: Background tasks untuk training dan simulation

## Security

1. **API Key Management**: Environment variables untuk sensitive credentials
2. **CORS Configuration**: Restrict origins in production
3. **Rate Limiting**: Implement rate limiting pada API endpoints
4. **Input Validation**: Pydantic models untuk request validation
