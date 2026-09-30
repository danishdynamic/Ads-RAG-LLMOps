# Backend Scripts

This directory contains utility scripts for data preparation, evaluation, experiment comparison, and development checks.

Run scripts from the `backend` directory using Python modules:

```bash
python -m scripts.<script_name>
```

## Scripts

### `evaluate_retrieval.py`

Runs the retrieval evaluation dataset and logs the results to MLflow.

Example:

```bash
python -m scripts.evaluate_retrieval --top-k 5
```

```bash
python -m scripts.evaluate_retrieval --top-k 10
```

Measures:

* Precision@K
* Recall@K
* Hit Rate@K
* Route Accuracy

The evaluation results are also logged as an MLflow artifact.

---

### `compare_rag_experiments.py`

Reads recorded MLflow retrieval experiments and prints a comparison of retrieval configurations.

Example:

```bash
python -m scripts.compare_rag_experiments
```

The current comparison examines:

```text
top_k = 5
top_k = 10
```

and reports:

* Precision
* Recall
* Hit Rate
* Route Accuracy

---

### `evaluate_answers.py`

Runs answer-generation evaluation using the RAG pipeline and Gemini-based LLM judge.

The judge evaluates:

* Faithfulness
* Relevance
* Completeness

Because answer evaluation requires LLM generation and judging calls, API quota and rate limits should be considered when running the full evaluation dataset.

---

### `test_ground_truth.py`

Checks the dynamic evaluation ground-truth logic against the database.

It is useful for verifying that evaluation expectations are generated from the current synthetic dataset.

---

## MLflow

Start the MLflow UI from the `backend` directory:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```

Experiment:

```text
ads-rag-llmops
```

## Typical Evaluation Workflow

```bash
# Retrieval experiments
python -m scripts.evaluate_retrieval --top-k 5

python -m scripts.evaluate_retrieval --top-k 10

# Compare experiments
python -m scripts.compare_rag_experiments

# Answer evaluation
python -m scripts.evaluate_answers
```

The scripts are intentionally kept separate from the application code so evaluation and experimentation can be run independently from the API service.
