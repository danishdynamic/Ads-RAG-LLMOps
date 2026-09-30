import json
import time  
from pathlib import Path

import mlflow

from app.db.database import SessionLocal
from app.evaluation.answer_evaluation import evaluate_answer
from app.evaluation.dataset import EVALUATION_DATASET

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)
mlflow.set_experiment("ads-rag-llmops-answer-evaluation")


def main():
    db = SessionLocal()
    results = []

    # Pace delay: 4 seconds delay = max 15 requests/minute
    DELAY_BETWEEN_CASES = 4.0

    try:
        mlflow.start_run(run_name="llm-answer-evaluation")
        mlflow.log_params(
            {
                "evaluation_cases": len(EVALUATION_DATASET),
                "top_k": 5,
                "judge_model": "gemini-3.1-flash-lite",
                "delay_between_cases_sec": DELAY_BETWEEN_CASES,
            }
        )

        print("=" * 70)
        print("ADS RAG ANSWER EVALUATION")
        print("=" * 70)

        for index, case in enumerate(
            EVALUATION_DATASET,
            start=1,
        ):
            print(f"\n[{index}/{len(EVALUATION_DATASET)}] {case.question}")

            result = evaluate_answer(
                db=db,
                question=case.question,
                top_k=5,
            )

            results.append(result)

            print(f"Route: {result['route']}")
            print(f"Faithfulness: {result['faithfulness']:.3f}")
            print(f"Relevance:    {result['relevance']:.3f}")
            print(f"Completeness: {result['completeness']:.3f}")

            # Pace out requests if there are remaining evaluation cases
            if index < len(EVALUATION_DATASET):
                time.sleep(DELAY_BETWEEN_CASES)

        # --------------------------------
        # Aggregation & Artifact Logging
        # --------------------------------
        if results:
            avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
            avg_relevance = sum(r["relevance"] for r in results) / len(results)
            avg_completeness = sum(r["completeness"] for r in results) / len(results)

            print("\n" + "=" * 70)
            print("OVERALL RESULTS")
            print("=" * 70)
            print(f"Faithfulness: {avg_faithfulness:.3f}")
            print(f"Relevance:    {avg_relevance:.3f}")
            print(f"Completeness: {avg_completeness:.3f}")

            mlflow.log_metrics(
                {
                    "avg_faithfulness": avg_faithfulness,
                    "avg_relevance": avg_relevance,
                    "avg_completeness": avg_completeness,
                }
            )

            report_path = PROJECT_ROOT / "answer_evaluation_report.json"
            with open(report_path, "w") as f:
                json.dump(results, f, indent=2)

            mlflow.log_artifact(str(report_path), artifact_path="evaluation")

        mlflow.end_run()

    finally:
        db.close()


if __name__ == "__main__":
    main()