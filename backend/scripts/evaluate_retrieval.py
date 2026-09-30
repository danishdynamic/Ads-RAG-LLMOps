import argparse
import json
from collections import defaultdict
from pathlib import Path

import mlflow

from app.core.config import get_settings
from app.db.database import SessionLocal
from app.evaluation.dataset import EVALUATION_DATASET
from app.evaluation.retrieval import evaluate_case
from app.observability.mlflow_tracker import configure_mlflow

settings = get_settings()
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG retrieval evaluation."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents/results to retrieve per case.",
    )
    args = parser.parse_args()

    db = SessionLocal()
    results = []

    try:
        configure_mlflow()

        with mlflow.start_run(
            run_name=f"retrieval-evaluation-top-k-{args.top_k}"
        ):
            # --------------------------------------------------
            # Self-describing Configuration Tracking
            # --------------------------------------------------
            mlflow.log_params(
                {
                    "top_k": args.top_k,
                    "embedding_model": settings.gemini_embeddings,
                    "generation_model": settings.gemini_model,
                    "retrieval_strategy": "sql_vector_hybrid",
                    "evaluation_dataset": "ads_evaluation_v1",
                    "evaluation_cases": len(EVALUATION_DATASET),
                }
            )

            print("=" * 70)
            print("ADS RAG RETRIEVAL EVALUATION")
            print("=" * 70)

            for index, case in enumerate(
                EVALUATION_DATASET,
                start=1,
            ):

                print()
                print(
                    f"[{index}/{len(EVALUATION_DATASET)}] "
                    f"{case.question}"
                )

                result = evaluate_case(
                    db=db,
                    case=case,
                    top_k=args.top_k,
                )

                results.append(result)

                print(
                    f"Route: "
                    f"{result['actual_route']} "
                    f"({'✓' if result['route_correct'] else '✗'})"
                )

                print(
                    f"Precision@{args.top_k}: "
                    f"{result['precision']:.3f}"
                )

                print(
                    f"Recall@{args.top_k}: "
                    f"{result['recall']:.3f}"
                )

                print(
                    f"Hit Rate@{args.top_k}: "
                    f"{result['hit_rate']:.3f}"
                )

            print()
            print("=" * 70)
            print("OVERALL RESULTS")
            print("=" * 70)

            if not results:
                return

            avg_precision = sum(
                result["precision"]
                for result in results
            ) / len(results)

            avg_recall = sum(
                result["recall"]
                for result in results
            ) / len(results)

            avg_hit_rate = sum(
                result["hit_rate"]
                for result in results
            ) / len(results)

            route_accuracy = sum(
                result["route_correct"]
                for result in results
            ) / len(results)

            print(
                f"Precision@{args.top_k}: {avg_precision:.3f}"
            )

            print(
                f"Recall@{args.top_k}:    {avg_recall:.3f}"
            )

            print(
                f"Hit Rate@{args.top_k}:  {avg_hit_rate:.3f}"
            )

            print(
                f"Route Acc:   {route_accuracy:.3f}"
            )

            # --------------------------------------------------
            # Dynamic MLflow Metric Logging
            # --------------------------------------------------
            mlflow.log_metrics(
                {
                    f"precision_at_{args.top_k}": avg_precision,
                    f"recall_at_{args.top_k}": avg_recall,
                    f"hit_rate_at_{args.top_k}": avg_hit_rate,

                    "avg_precision": avg_precision,
                    "avg_recall": avg_recall,
                    "avg_hit_rate": avg_hit_rate,

                    "route_accuracy": route_accuracy,
                }
            )

            print()
            print("=" * 70)
            print("BY QUERY TYPE")
            print("=" * 70)

            grouped = defaultdict(list)

            for result in results:
                grouped[
                    result["query_type"]
                ].append(result)

            for query_type, group in grouped.items():

                precision = sum(
                    result["precision"]
                    for result in group
                ) / len(group)

                recall = sum(
                    result["recall"]
                    for result in group
                ) / len(group)

                hit_rate = sum(
                    result["hit_rate"]
                    for result in group
                ) / len(group)

                print()
                print(query_type.upper())

                print(
                    f"  Precision@{args.top_k}: {precision:.3f}"
                )

                print(
                    f"  Recall@{args.top_k}:    {recall:.3f}"
                )

                print(
                    f"  Hit Rate@{args.top_k}:  {hit_rate:.3f}"
                )

                # Dynamic per-query-type metrics
                mlflow.log_metrics(
                    {
                        f"{query_type}_precision_at_{args.top_k}": precision,
                        f"{query_type}_recall_at_{args.top_k}": recall,
                        f"{query_type}_hit_rate_at_{args.top_k}": hit_rate,
                    }
                )

            # Save and log evaluation report artifact
            report_path = PROJECT_ROOT / "evaluation_report.json"
            with open(report_path, "w") as f:
                json.dump(
                    results,
                    f,
                    indent=2,
                )

            mlflow.log_artifact(
                str(report_path),
                artifact_path="evaluation",
            )

    finally:
        db.close()


if __name__ == "__main__":
    main()