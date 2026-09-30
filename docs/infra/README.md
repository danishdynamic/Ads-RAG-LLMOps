# Infrastructure & Local Development Guide

This guide covers setting up, running, and managing the local containerized environment for the project, including the PostgreSQL database with `pgvector` support, the FastAPI backend, and MLflow experiment tracking.

---

## 🏗️ Architecture Overview

| Service | Container Name | Image / Build Context | Internal Port | Host Port | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Database** | `ads-llmops-postgres` | `pgvector/pgvector:pg16` | `5432` | `5433` | Vector store & app database |
| **MLflow Server** | `ads-llmops-mlflow` | `ghcr.io/mlflow/mlflow:v2.11.3` | `5000` | `5000` | RAG & LLM tracking server |
| **Backend API** | `ads-llmops-backend` | `./backend` (`python:3.12`) | `8000` | `8000` | FastAPI application layer |

---

## 🚀 Quick Start

### 1. Prerequisites
- [Docker Engine](https://docs.docker.com/get-docker/) `v24.0+`
- [Docker Compose](https://docs.docker.com/compose/) `v2.20+`

### 2. Environment Variables
Create a `.env` file in the project root with the following default credentials:

```env
POSTGRES_USER=ads_user
POSTGRES_PASSWORD=ads_password
POSTGRES_DB=ads_llmops
DATABASE_URL=postgresql://ads_user:ads_password@postgres:5432/ads_llmops
MLFLOW_TRACKING_URI=http://mlflow:5000
```

3. Spin Up Services
To build and start all containers in detached mode:

```Bash
docker compose up -d --build
```
To verify all containers are healthy:

```Bash
docker compose ps
```

## 🐋 Complete Docker Compose Reference (docker-compose.yml)

```YAML
version: "3.8"

services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: ads-llmops-postgres
    restart: always
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-ads_user}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-ads_password}
      POSTGRES_DB: ${POSTGRES_DB:-ads_llmops}
    ports:
      - "5433:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-ads_user} -d ${POSTGRES_DB:-ads_llmops}"]
      interval: 5s
      timeout: 5s
      retries: 5

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.11.3
    container_name: ads-llmops-mlflow
    restart: always
    ports:
      - "5000:5000"
    environment:
      - BACKEND_STORE_URI=postgresql://${POSTGRES_USER:-ads_user}:${POSTGRES_PASSWORD:-ads_password}@postgres:5432/${POSTGRES_DB:-ads_llmops}
    command: >
      mlflow server
      --backend-store-uri postgresql://${POSTGRES_USER:-ads_user}:${POSTGRES_PASSWORD:-ads_password}@postgres:5432/${POSTGRES_DB:-ads_llmops}
      --default-artifact-root /mlflow/artifacts
      --host 0.0.0.0
      --port 5000
    depends_on:
      postgres:
        condition: service_healthy
    volumes:
      - mlflow_artifacts:/mlflow/artifacts

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: ads-llmops-backend
    restart: always
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://${POSTGRES_USER:-ads_user}:${POSTGRES_PASSWORD:-ads_password}@postgres:5432/${POSTGRES_DB:-ads_llmops}
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      postgres:
        condition: service_healthy
      mlflow:
        condition: service_started
    volumes:
      - ./backend:/app

volumes:
  postgres_data:
  mlflow_artifacts:

```

## 🛠️ Connection Endpoints
- FastAPI API Documentation: ```http://localhost:8000/docs```

- MLflow Tracking UI: ```http://localhost:5000```

- PostgreSQL Connection (Host Machine): ```postgresql://ads_user:ads_password@localhost:5433/ads_llmops```

- PostgreSQL Connection (Inside Docker Network): ```postgresql://ads_user:ads_password@postgres:5432/ads_llmops```

## 🧪 Enabling pgvector in PostgreSQL
To enable vector operations, verify or run the following SQL command on startup inside your application migration script or SQLAlchemy startup hook:

```SQL
CREATE EXTENSION IF NOT EXISTS vector;
```

## 🧹 Maintenance & Troubleshooting
View Container Logs:


```Bash
docker compose logs -f backend
docker compose logs -f mlflow
```

Access Database CLI directly:

```Bash
docker exec -it ads-llmops-postgres psql -U ads_user -d ads_llmops
```
Reset Persisted Data & Volumes:

```Bash
docker compose down -v
```