import mlflow

from app.observability.mlflow_tracker import configure_mlflow


def main():
    configure_mlflow()

    experiment = mlflow.get_experiment_by_name("ads-rag-llmops")

    if experiment is None:
        print("Experiment not found.")
        return

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    print("\nMLFLOW RUNS")
    print("=" * 80)

    for _, run in runs.iterrows():
        print(
            f"run_name={run.get('tags.mlflow.runName')}"
        )
        print(
            f"run_id={run.get('run_id')}"
        )
        print(
            f"status={run.get('status')}"
        )
        print(
            f"top_k={run.get('params.top_k')}"
        )
        print("-" * 80)


if __name__ == "__main__":
    main()