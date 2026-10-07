/**
 * API Client for Aegis-JKN ML/DL Backend
 */
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface PredictionRequest {
  claim_data: Record<string, any>;
}

export interface PredictionResponse {
  status: string;
  fraud_probability: number;
  risk_level: string;
  channel_contributions: {
    topological: number;
    feature: number;
    semantic: number;
    shared: number;
  };
}

export interface SimulationRequest {
  num_agents?: number;
  num_steps?: number;
  simulation_type?: string;
}

export interface SimulationResponse {
  status: string;
  simulation_results: any;
  comparison_with_baseline: any;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async get(endpoint: string) {
    const response = await fetch(`${this.baseUrl}${endpoint}`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  }

  async post(endpoint: string, data: any) {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  }

  // Health check
  async healthCheck() {
    return this.get('/health');
  }

  // Data pipeline
  async fetchData() {
    return this.post('/api/data/fetch', {});
  }

  // Model training
  async trainModel(epochs: number = 100, batchSize: number = 32, learningRate: number = 0.001) {
    return this.post('/api/model/train', {
      epochs,
      batch_size: batchSize,
      learning_rate: learningRate,
    });
  }

  // Prediction
  async predictFraud(claimData: PredictionRequest): Promise<PredictionResponse> {
    return this.post('/api/model/predict', claimData);
  }

  // OASIS simulation
  async runSimulation(request: SimulationRequest): Promise<SimulationResponse> {
    return this.post('/api/oasis/simulate', request);
  }

  // Metrics
  async getMetrics() {
    return this.get('/api/metrics');
  }
}

export const apiClient = new ApiClient();
