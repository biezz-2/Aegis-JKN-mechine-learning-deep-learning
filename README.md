# Aegis-JKN Machine Learning & Deep Learning Connector

Backend ML/DL service dan OASIS integration untuk platform deteksi fraud JKN (Jaminan Kesehatan Nasional).

## 🎯 Overview

Proyek ini menyediakan konektor all-in-one yang menghubungkan:
- **Data Pipeline**: Ekstraksi dataset dari Notion API
- **ML/DL Models**: Implementasi MHGSL (Multi-Channel Heterogeneous Graph Structure Learning)
- **OASIS Integration**: Simulasi agent-based untuk analisis fraud pattern
- **Real-time Monitoring**: WebSocket API untuk dashboard monitoring di Vercel
- **Database**: Redis + PostgreSQL untuk caching dan persistence

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Aegis-JKN Vercel Dashboard                   │
│              (https://aegisjkn.vercel.app/)                      │
└────────────────────────┬────────────────────────────────────────┘
                         │ WebSocket
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                 FastAPI Backend Server                           │
│              (http://localhost:8000)                            │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Data Pipeline│  │ MHGSL Model │  │ OASIS Sim  │             │
│  │  (Notion)   │  │  (PyTorch)  │  │ (OpenAI)   │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                      │
│         └────────────────┼────────────────┘                      │
│                          │                                       │
│                    ┌─────▼─────┐                                │
│                    │ Redis +   │                                │
│                    │ PostgreSQL│                                │
│                    └───────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Project Structure

```
Aegis-JKN-mechine-learning-deep-learning/
├── backend/                      # ML/DL backend service
│   ├── src/
│   │   ├── data/                # Data pipeline
│   │   │   ├── notion_client.py # Notion API client
│   │   │   └── pipeline.py      # ETL & graph construction
│   │   ├── models/              # ML/DL models
│   │   │   └── mhgsl.py         # MHGSL implementation
│   │   ├── oasis/               # OASIS integration
│   │   │   ├── agent_mapper.py  # JKN → OASIS mapping
│   │   │   └── simulation.py    # Fraud simulation
│   │   ├── api/                 # FastAPI endpoints
│   │   │   └── main.py          # API server
│   │   └── utils/               # Utilities
│   │       └── config.py        # Configuration
│   ├── requirements.txt
│   └── .env.example
├── frontend-dashboard/           # Dashboard components for Vercel
│   └── src/
│       ├── components/
│       │   ├── ml-monitor/      # ML training monitor
│       │   ├── oasis-viz/       # OASIS simulation visualizer
│       │   └── real-time/       # Real-time metrics
│       └── lib/
│           └── api-client.ts    # Backend API client
├── docs/                        # Documentation
│   ├── architecture.md
│   ├── api-reference.md
│   ├── oasis-integration.md
│   └── deployment-guide.md
├── notebooks/                   # Jupyter notebooks
├── datasets/                     # Sample datasets
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- PostgreSQL 14+
- Redis 7+
- Node.js 18+ (untuk frontend dashboard)

### 1. Clone Repository

```bash
git clone https://github.com/biezz-2/Aegis-JKN-mechine-learning-deep-learning.git
cd Aegis-JKN-mechine-learning-deep-learning
```

### 2. Setup Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

### 3. Setup Database

```bash
# PostgreSQL
createdb aegis_jkn_ml

# Redis (Docker)
docker run -d -p 6379:6379 redis:7-alpine
```

### 4. Run Backend Server

```bash
cd backend/src
python -m api.main
```

Server akan berjalan di `http://localhost:8000`

### 5. Integrate dengan Aegis-JKN Vercel

Copy file dari `frontend-dashboard/` ke repo Aegis-JKN:

```bash
# Copy components
cp -r frontend-dashboard/src/components/* /path/to/Aegis-JKN/src/components/

# Copy API client
cp frontend-dashboard/src/lib/api-client.ts /path/to/Aegis-JKN/src/lib/
```

## 📊 API Endpoints

### Health Check
```bash
GET /health
```

### Data Pipeline
```bash
POST /api/data/fetch
```
Fetch data dari Notion dan siapkan untuk ML pipeline.

### Model Training
```bash
POST /api/model/train
Content-Type: application/json

{
  "epochs": 100,
  "batch_size": 32,
  "learning_rate": 0.001
}
```

### Fraud Prediction
```bash
POST /api/model/predict
Content-Type: application/json

{
  "claim_data": {
    "patient_id": "P001",
    "doctor_id": "D001",
    "procedure": "PROC_001",
    "diagnosis": "DIAG_001",
    "amount": 1000000
  }
}
```

### OASIS Simulation
```bash
POST /api/oasis/simulate
Content-Type: application/json

{
  "num_agents": 100,
  "num_steps": 10,
  "simulation_type": "all"
}
```

### WebSocket (Real-time Monitoring)
```bash
WS /ws
```

## 🔧 Configuration

Environment variables di `.env`:

```env
# Notion API
NOTION_TOKEN=your_notion_integration_token
NOTION_PAGE_ID=3f23e603-1ab7-808d-b2cb-fe92cd62391d
NOTION_DATASET_DB_ID=310b35a5-9346-42df-af47-df9897cdcda1
NOTION_CLAIMS_DB_ID=c783afb3-d0d4-4582-9545-6180b7b7756b

# OpenAI API (OASIS)
OPENAI_API_BASE=https://9router.biezz.my.id/v1
OPENAI_API_KEY=sk-your-api-key
OPENAI_MODEL=oasis

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/aegis_jkn_ml
REDIS_URL=redis://localhost:6379/0

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# ML/DL Configuration
GCN_HIDDEN_DIM=128
GCN_NUM_LAYERS=3
ATTENTION_HEADS=4
BATCH_SIZE=32
LEARNING_RATE=0.001
EPOCHS=100

# OASIS Configuration
OASIS_NUM_AGENTS=100
OASIS_SIMULATION_STEPS=10
OASIS_ACTIVATION_PROBABILITY=0.1
```

## 🧪 Testing

### Test Data Pipeline
```bash
cd backend/src
python -c "from data.pipeline import MHGSLDataPipeline; pipeline = MHGSLDataPipeline(); data = pipeline.run_pipeline(); print(data)"
```

### Test OASIS Simulation
```bash
cd backend/src
python -c "import asyncio; from oasis.simulation import JKNFraudSimulation; sim = JKNFraudSimulation('https://9router.biezz.my.id/v1', 'sk-your-key'); asyncio.run(sim.run_all_simulations(50, 5))"
```

### Test API
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/data/fetch
```

## 📈 Performance Metrics

Target metrics:
- **MHGSL Model AUPRC**: ≥0.91
- **OASIS Detection Rate**: ≥0.85
- **API Response Time**: <500ms p95
- **WebSocket Latency**: <1s

## 🤝 Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - lihat [LICENSE](LICENSE) untuk detail

## 👥 Authors

- **Syarif Attabi** - Initial work - [biezz-2](https://github.com/biezz-2)

## 🙏 Acknowledgments

- BPJS Kesehatan untuk dataset dan use case
- CAMEL-AI untuk platform OASIS
- Aegis-JKN team untuk arsitektur MHGSL

## 📞 Support

Untuk pertanyaan atau support, hubungi:
- Email: biezzpanel@gmail.com
- GitHub Issues: [Create Issue](https://github.com/biezz-2/Aegis-JKN-mechine-learning-deep-learning/issues)
