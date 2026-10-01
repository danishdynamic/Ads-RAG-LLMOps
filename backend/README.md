# Ads RAG LLMOps — Backend

> The backend implements the API, database layer, hybrid RAG pipeline, evaluation framework, MLflow tracking, and observability components for the Ads RAG LLMOps project.

## Backend Responsibilities

The backend provides:

* FastAPI REST API
* PostgreSQL persistence
* pgvector semantic search
* Synthetic advertising data generation
* Gemini embeddings
* SQL retrieval
* Vector retrieval
* Hybrid retrieval
* Query intent extraction
* Gemini answer generation
* Retrieval evaluation
* LLM-as-a-judge evaluation
* MLflow experiment tracking
* RAG latency and retrieval observability

## Architecture

```mermaid
graph TD
    classDef api fill:#2563eb,color:#fff,stroke:#1d4ed8,stroke-width:2px;
    classDef orchestrator fill:#7c3aed,color:#fff,stroke:#6d28d9,stroke-width:2px;
    classDef storage fill:#059669,color:#fff,stroke:#047857,stroke-width:2px;
    classDef gen fill:#d97706,color:#fff,stroke:#b45309,stroke-width:2px;
    classDef obs fill:#475569,color:#fff,stroke:#334155,stroke-width:2px;

    API["FastAPI Endpoint"]:::api
    
    subgraph RAG ["RAG Orchestration"]
        Pipe["RAG Pipeline"]:::orchestrator
        CB["Context Builder"]:::orchestrator
    end

    subgraph DataStores ["Hybrid Retrieval Layer"]
        direction LR
        subgraph SQLPath ["SQL Route"]
            SR["SQL Retriever"]:::storage --> PG[("PostgreSQL")]:::storage
        end
        subgraph VectorPath ["Vector Route"]
            VR["Vector Retriever"]:::storage --> PGV[("pgvector")]:::storage
        end
    end

    GG["Gemini Generator"]:::gen
    FA["Final Answer"]:::gen
    ML["MLflow Tracking"]:::obs

    API --> Pipe
    Pipe --> SR
    Pipe --> VR
    PG --> CB
    PGV --> CB
    CB --> GG
    GG --> FA
    FA --> ML
```

## Requirements

* Python 3.11+
* Docker
* PostgreSQL with pgvector
* Gemini API key

## Environment Variables

Create `.env` from `.env.example`.

```env
APP_NAME=Ads RAG LLMOps
APP_ENV=development
DEBUG=true

DATABASE_URL=postgresql+psycopg://ads_user:ads_password@localhost:5432/ads_llmops

GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.1-flash-lite
GEMINI_EMBEDDINGS=gemini-embedding-001

MLFLOW_TRACKING_URI=http://localhost:5000
```

## Database

The project uses PostgreSQL with the pgvector extension.

Start the database from the project root:

```bash
docker compose up -d
```

The database is configured as:

```text
Database: ads_llmops
User: ads_user
Port: 5432
```

## Migrations

Run:

```bash
alembic upgrade head
```

The migration enables the pgvector extension and creates the application tables.

## Data Model

The main entities are:

### Campaign

Stores campaign level information.

### Ad

Stores structured advertising performance and marketing information.

Important fields include:

```text
platform
date
impressions
reach
clicks
spend
conversions
conversion_value
ctr
cpc
cpa
roas
audience
age_group
gender
country
device
placement
headline
primary_text
description
call_to_action
marketing_angle
tone
offer_type
product_category
```

### AdDocument

Stores the semantic representation of an ad.

The document contains:

* Ad content
* Embedding vector
* Reference to the associated ad

Embeddings use a 768-dimensional pgvector column.

## Synthetic Data

The project generates:

```text
50 campaigns
1,000 ads
1,000 ad documents
```

The data covers Google and Facebook advertising.

The generated ads contain realistic marketing copy and controlled performance relationships.

## Embedding Pipeline

Marketing content is converted into embeddings using:

```text
gemini-embedding-001
```

Embeddings are generated in batches and stored in PostgreSQL using pgvector.

The vector search uses cosine distance to retrieve semantically related ads.

## Retrieval

The RAG pipeline supports three routes.

### SQL

Structured analytical questions are handled using PostgreSQL queries.

Examples:

```text
Which ad had the highest ROAS?
Which ad had the lowest CPA?
What is the average ROAS by marketing angle?
```

### Vector

Semantic questions are handled through pgvector.

Examples:

```text
Which ads use urgency messaging?
Find ads using premium messaging.
Which ads use discount focused language?
```

### Hybrid

Questions requiring both structured filtering and semantic understanding use both retrieval methods.

Example:

```text
Find high performing Facebook ads that use urgency messaging.
```

The pipeline can combine:

* Platform filters
* ROAS thresholds
* Marketing angles
* Semantic similarity

## Query Routing

The router classifies incoming questions into:

```text
sql
vector
hybrid
```

Intent extraction identifies structured constraints such as:

```text
platform
marketing_angle
min_roas
max_roas
min_ctr
max_cpa
```

The extracted intent is logged as an MLflow artifact for observability.

## Context Construction

SQL and vector results are converted into context before being passed to Gemini.

The context contains the retrieved advertising information and, where applicable, the original marketing copy.

The system also removes duplicate ads when they appear in both SQL and vector results.

## Answer Generation

The generation model is:

```text
gemini-3.1-flash-lite
```

Generation uses a low temperature and a constrained instruction that requires the model to answer from the supplied advertising data.

The model is instructed not to invent:

* Ads
* Metrics
* Campaigns
* Conclusions

## API

Start the backend:

```bash
uvicorn app.main:app --reload
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### Health

```text
GET /health
```

### Database Health

```text
GET /health/db
```

### RAG Query

```text
POST /api/v1/rag/query
```

Example request:

```json
{
  "question": "Find high-performing Facebook ads that use urgency messaging.",
  "top_k": 5
}
```

The response contains:

* Question
* Route
* Extracted intent
* Generated answer
* Retrieval context
* Retrieval results
* Observability metadata

## Observability

RAG requests track:

```text
top_k
retrieved_document_count
retrieval_latency_ms
generation_latency_ms
total_latency_ms
embedding_model
generation_model
retrieval_strategy
environment
```

MLflow artifacts include:

```text
query_intent.json
retrieval_results.json
retrieval_context.txt
generated_answer.txt
```

## Evaluation

The evaluation framework contains 15 test cases covering:

* SQL
* Semantic
* Hybrid

Retrieval metrics:

```text
Precision@K
Recall@K
Hit Rate@K
Route Accuracy
```

Evaluation results are grouped by query type as well as overall.

## Answer Evaluation

Generated answers can be evaluated with an LLM judge.

The judge evaluates:

```text
Faithfulness
Relevance
Completeness
```

Each score ranges from:

```text
0.0 → 1.0
```

The judge also provides a short reason for its evaluation.

## MLflow

MLflow is configured with a SQLite backend:

```text
sqlite:///mlflow.db
```

Start the UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open:

```text
http://127.0.0.1:5000
```

Experiment:

```text
ads-rag-llmops
```

### Retrieval Experiments

The project evaluates different retrieval depths:

```bash
python -m scripts.evaluate_retrieval --top-k 5
```

```bash
python -m scripts.evaluate_retrieval --top-k 10
```

Comparison:

```bash
python -m scripts.compare_rag_experiments
```

## Project Structure

```text
backend/
├── README.md
├── requirements.txt
├── alembic.ini
├── .env
├── .env.example
├── alembic/
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
├── app/
│   ├── api/
│   │   ├── health.py
│   │   └── rag.py
│   ├── core/
│   │   └── config.py
│   ├── db/
│   │   ├── database.py
│   │   └── models.py
│   ├── evaluation/
│   │   ├── dataset.py
│   │   ├── schemas.py
│   │   ├── metrics.py
│   │   ├── ground_truth.py
│   │   ├── retrieval.py
│   │   ├── llm_judge.py
│   │   └── answer_evaluation.py
│   ├── observability/
│   │   └── mlflow_tracker.py
│   └── rag/
│       ├── context.py
│       ├── generator.py
│       ├── intent.py
│       ├── pipeline.py
│       ├── router.py
│       ├── schemas.py
│       ├── sql_retriever.py
│       └── vector_retriever.py
└── scripts/
    ├── README.md
    ├── evaluate_retrieval.py
    ├── compare_rag_experiments.py
    ├── evaluate_answers.py
    └── test_ground_truth.py
```

## Development Workflow

Typical local workflow:

```bash
# Start database
docker compose up -d

# Run migrations
alembic upgrade head

# Start API
uvicorn app.main:app --reload

# Run retrieval evaluation
python -m scripts.evaluate_retrieval --top-k 5

# Run another retrieval experiment
python -m scripts.evaluate_retrieval --top-k 10

# Compare experiments
python -m scripts.compare_rag_experiments

# Start MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Design Goal

The backend is designed as an LLMOps oriented RAG system rather than a simple question-answering API.

The important engineering loop is:

```mermaid
graph TD
    classDef step fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#fff;
    classDef loop fill:#1e1b4b,stroke:#8b5cf6,stroke-width:2px,color:#fff;

    R["1. Retrieve<br/><i>(Hybrid SQL + Vector Search)</i>"]:::step
    G["2. Generate<br/><i>(Gemini / Prompt Execution)</i>"]:::step
    E["3. Evaluate<br/><i>(Precision, Recall, Hit Rate)</i>"]:::step
    T["4. Track<br/><i>(MLflow Params & Artifacts)</i>"]:::step
    O["5. Observe<br/><i>(Traces & Latency Metrics)</i>"]:::step
    C["6. Compare<br/><i>(Experiment Runs & Top-K)</i>"]:::step
    I["7. Iterate<br/><i>(Tune Prompts, Chunking, Routing)</i>"]:::loop

    R --> G --> E --> T --> O --> C --> I
    I -. "Refine & Rerun" .-> R
```
