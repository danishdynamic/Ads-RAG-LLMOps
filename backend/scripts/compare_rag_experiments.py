import mlflow

from app.observability.mlflow_tracker import configure_mlflow


EXPERIMENT_NAME = "ads-rag-llmops"


def main():
    configure_mlflow()

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{EXPERIMENT_NAME}' was not found."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "tags.mlflow.runName LIKE "
            "'retrieval-evaluation-top-k-%'"
        ),
        order_by=["start_time DESC"],
    )

    latest_runs = {}

    for _, run in runs.iterrows():
        top_k = run.get("params.top_k")

        if top_k is None:
            continue

        if top_k not in latest_runs:
            latest_runs[top_k] = run

    print("\nRAG EXPERIMENT COMPARISON")
    print("=" * 80)

    for top_k in sorted(latest_runs):
        run = latest_runs[top_k]

        if top_k == "5":
            precision = run.get("metrics.precision_at_5")
            recall = run.get("metrics.recall_at_5")
            hit_rate = run.get("metrics.hit_rate_at_5")

        elif top_k == "10":
            precision = run.get("metrics.precision_at_10")
            recall = run.get("metrics.recall_at_10")
            hit_rate = run.get("metrics.hit_rate_at_10")

        else:
            continue

        route_accuracy = run.get(
            "metrics.route_accuracy"
        )

        print(f"\ntop_k = {top_k}")
        print(f"Precision:      {precision:.3f}")
        print(f"Recall:         {recall:.3f}")
        print(f"Hit Rate:       {hit_rate:.3f}")
        print(f"Route Accuracy: {route_accuracy:.3f}")

    print("\n")
    print("Interpretation")
    print("-" * 80)
    print(
        "The experiments measure the effect of changing "
        "retrieval depth while keeping the evaluation "
        "dataset and retrieval strategy consistent."
    )
    print(
        "The metrics describe observed retrieval behavior "
        "for each configuration."
    )


if __name__ == "__main__":
    main()