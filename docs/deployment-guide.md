# Deployment Guide

## Local Development

### Prerequisites

- Python 3.9+
- PostgreSQL 14+
- Redis 7+
- Git

### Setup

1. **Clone Repository**
```bash
git clone https://github.com/biezz-2/Aegis-JKN-mechine-learning-deep-learning.git
cd Aegis-JKN-mechine-learning-deep-learning
```

2. **Setup Python Environment**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Setup PostgreSQL**
```bash
# Create database
createdb aegis_jkn_ml

# Or using psql
psql -U postgres
CREATE DATABASE aegis_jkn_ml;
\q
```

4. **Setup Redis**
```bash
# Using Docker
docker run -d -p 6379:6379 --name redis redis:7-alpine

# Or install Redis directly
# macOS: brew install redis
# Ubuntu: sudo apt-get install redis-server
# Windows: Download from https://redis.io/download
```

5. **Configure Environment**
```bash
cd backend
cp .env.example .env
# Edit .env with your credentials
```

6. **Run Backend Server**
```bash
cd backend/src
python -m api.main
```

Server will start at `http://localhost:8000`

## Production Deployment

### Option 1: Docker Deployment

1. **Create Dockerfile**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/src ./src

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. **Build and Run**
```bash
docker build -t aegis-jkn-ml .
docker run -d -p 8000:8000 --env-file .env aegis-jkn-ml
```

### Option 2: Railway Deployment

1. **Install Railway CLI**
```bash
npm install -g @railway/cli
```

2. **Login and Deploy**
```bash
railway login
railway init
railway up
```

3. **Add Environment Variables**
```bash
railway variables set NOTION_TOKEN=your_token
railway variables set OPENAI_API_KEY=your_key
# ... add other variables
```

### Option 3: DigitalOcean App Platform

1. **Push to GitHub**
2. **Create App in DigitalOcean**
3. **Connect GitHub repository**
4. **Configure build settings**
5. **Add environment variables**

## Environment Variables

Required environment variables:

```env
# Notion API
NOTION_TOKEN=your_notion_integration_token
NOTION_PAGE_ID=3f23e603-1ab7-808d-b2cb-fe92cd62391d
NOTION_DATASET_DB_ID=310b35a5-9346-42df-af47-df9897cdcda1
NOTION_CLAIMS_DB_ID=c783afb3-d0d4-4582-9545-6180b7b7756b

# OpenAI API
OPENAI_API_BASE=https://9router.biezz.my.id/v1
OPENAI_API_KEY=sk-your-api-key
OPENAI_MODEL=oasis

# Database
DATABASE_URL=postgresql://user:password@host:5432/aegis_jkn_ml
REDIS_URL=redis://host:6379/0

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
```

## Monitoring

### Health Check
```bash
curl http://your-domain.com/health
```

### Logs
```bash
# Docker
docker logs aegis-jkn-ml

# Railway
railway logs

# DigitalOcean
# Check logs in App Platform dashboard
```

### Metrics
```bash
curl http://your-domain.com/api/metrics
```

## Troubleshooting

### PostgreSQL Connection Error
- Check DATABASE_URL format
- Ensure PostgreSQL is running
- Verify database exists

### Redis Connection Error
- Check Redis is running: `redis-cli ping`
- Verify REDIS_URL format
- Check firewall settings

### Notion API Error
- Verify NOTION_TOKEN is valid
- Check Notion integration has access to databases
- Ensure database IDs are correct

### OASIS Simulation Error
- Verify OpenAI API key
- Check API base URL
- Ensure model name is correct

## Scaling

### Horizontal Scaling
```bash
# Run multiple workers
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Database Pooling
```python
# Add to database configuration
SQLALCHEMY_ENGINE_OPTIONS = {
    "pool_size": 20,
    "max_overflow": 10,
    "pool_pre_ping": True
}
```

### Redis Clustering
```bash
# Setup Redis cluster
redis-cli --cluster create 127.0.0.1:7000 127.0.0.1:7001 ...
```

## Security

1. **Use HTTPS** in production
2. **Restrict CORS** origins
3. **Implement rate limiting**
4. **Use secrets management** (e.g., AWS Secrets Manager, HashiCorp Vault)
5. **Enable authentication** (JWT or API keys)
6. **Regular security updates**

## Backup

### PostgreSQL Backup
```bash
pg_dump aegis_jkn_ml > backup.sql
```

### Redis Backup
```bash
redis-cli BGSAVE
```

### Restore
```bash
psql aegis_jkn_ml < backup.sql
```
