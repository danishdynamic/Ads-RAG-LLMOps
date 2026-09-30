import time
from contextlib import contextmanager
from pathlib import Path

import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"
MLFLOW_EXPERIMENT = "ads-rag-llmops"


def configure_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


@contextmanager
def track_rag_run(
    question: str,
    route: str,
    model_name: str,
):
    configure_mlflow()

    # Reuse an existing MLflow run if one already exists.
    if mlflow.active_run() is not None:
        start_time = time.perf_counter()

        mlflow.log_params(
            {
                "question": question,
                "route": route,
                "model_name": model_name,
            }
        )

        metadata = {
            "run_id": mlflow.active_run().info.run_id,
            "question": question,
            "route": route,
            "model_name": model_name,
        }

        yield metadata

        total_latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        mlflow.log_metric(
            "rag_total_latency_ms",
            total_latency_ms,
        )

        return

    # Otherwise create a new run.
    with mlflow.start_run() as run:
        start_time = time.perf_counter()

        mlflow.log_params(
            {
                "question": question,
                "route": route,
                "model_name": model_name,
            }
        )

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
                "rag_total_latency_ms",
                total_latency_ms,
            )