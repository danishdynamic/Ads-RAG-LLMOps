import mlflow
from sqlalchemy.orm import Session

from app.evaluation.llm_judge import judge_answer
from app.rag.pipeline import answer_query


def evaluate_answer(
    db: Session,
    question: str,
    top_k: int = 5,
) -> dict:

    result = answer_query(
        db=db,
        query=question,
        top_k=top_k,
    )

    evaluation = judge_answer(
        question=question,
        context=result["context"],
        answer=result["answer"],
    )

    mlflow.log_metrics(
        {
            "faithfulness": evaluation["faithfulness"],
            "relevance": evaluation["relevance"],
            "completeness": evaluation["completeness"],
        }
    )

    return {
        "question": question,
        "route": result["route"],
        "answer": result["answer"],
        "faithfulness": evaluation["faithfulness"],
        "relevance": evaluation["relevance"],
        "completeness": evaluation["completeness"],
        "reason": evaluation["reason"],
    }