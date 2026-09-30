import json
import time
from contextlib import contextmanager
from pathlib import Path

from pathlib import Path

import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)

mlflow.set_experiment("ads-rag-llmops")


@contextmanager
def track_rag_run(
    question: str,
    route: str,
    model_name: str,
):
    with mlflow.start_run() as run:
        start_time = time.perf_counter()

        mlflow.log_param("question", question)
        mlflow.log_param("route", route)
        mlflow.log_param("model_name", model_name)

        metadata = {
            "run_id": run.info.run_id,
            "question": question,
            "route": route,
            "model_name": model_name,
        }

        try:
            yield metadata

        finally:
            total_latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            mlflow.log_metric(
                "total_latency_ms",
                total_latency_ms,
            )