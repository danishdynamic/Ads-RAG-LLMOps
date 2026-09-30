import time
import mlflow

from sqlalchemy.orm import Session

from app.rag.context import ad_to_context
from app.rag.generator import generate_answer
from app.rag.intent import extract_intent
from app.rag.router import classify_query
from app.rag.sql_retriever import (
    average_performance_by_angle,
    filtered_ads,
    highest_roas,
    lowest_cpa,
)
from app.rag.vector_retriever import vector_search
from app.core.config import get_settings
from app.observability.mlflow_tracker import track_rag_run

settings = get_settings()


def deduplicate_results(results: dict) -> dict:
    sql_ad_ids = set()

    for item in results["sql_results"]:
        if isinstance(item, str):
            marker = "Ad ID: "
            if marker in item:
                ad_id = item.split(marker)[1].split("\n")[0]
                sql_ad_ids.add(ad_id)

    results["vector_results"] = [
        item
        for item in results["vector_results"]
        if item["ad_id"] not in sql_ad_ids
    ]

    return results


def retrieve(
    db: Session,
    query: str,
    top_k: int = 5,
):
    route = classify_query(query)
    intent = extract_intent(query)

    results = {
        "route": route,
        "intent": intent,
        "sql_results": [],
        "vector_results": [],
    }

    # --------------------------------------------------
    # SQL
    # --------------------------------------------------

    if route in {"sql", "hybrid"}:
        query_lower = query.lower()

        if "highest" in query_lower and "roas" in query_lower:
            ads = highest_roas(
                db=db,
                platform=intent["platform"],
                top_k=top_k,
            )
            results["sql_results"] = [ad_to_context(ad) for ad in ads]

        elif "lowest" in query_lower and "cpa" in query_lower:
            ads = lowest_cpa(
                db=db,
                platform=intent["platform"],
                top_k=top_k,
            )
            results["sql_results"] = [ad_to_context(ad) for ad in ads]

        elif "average" in query_lower:
            rows = average_performance_by_angle(db)
            results["sql_results"] = [
                {
                    "marketing_angle": row.marketing_angle,
                    "ad_count": row.ad_count,
                    "avg_ctr": float(row.avg_ctr),
                    "avg_roas": float(row.avg_roas),
                    "avg_cpa": float(row.avg_cpa),
                }
                for row in rows
            ]

        else:
            ads = filtered_ads(
                db=db,
                platform=intent["platform"],
                marketing_angle=intent["marketing_angle"],
                min_roas=intent["min_roas"],
                max_roas=intent["max_roas"],
                min_ctr=intent["min_ctr"],
                max_cpa=intent["max_cpa"],
                top_k=top_k * 3,
            )
            results["sql_results"] = [ad_to_context(ad) for ad in ads]

    # --------------------------------------------------
    # Vector search
    # --------------------------------------------------

    if route in {"vector", "hybrid"}:
        vector_results = vector_search(
            db=db,
            query=query,
            top_k=top_k,
        )

        results["vector_results"] = [
            {
                "ad_id": ad.ad_id,
                "distance": float(distance),
                "content": document.content,
            }
            for ad, document, distance in vector_results
        ]

    return results


def build_context(results: dict) -> str:
    sections = []

    if results["sql_results"]:
        sections.append(
            "STRUCTURED SQL RESULTS:\n" + str(results["sql_results"])
        )

    if results["vector_results"]:
        sections.append(
            "SEMANTIC SEARCH RESULTS:\n"
            + "\n\n".join(
                item["content"] for item in results["vector_results"]
            )
        )

    return "\n\n".join(sections)


def answer_query(
    db: Session,
    query: str,
    top_k: int = 5,
):
    retrieval_start = time.perf_counter()

    retrieval = retrieve(
        db=db,
        query=query,
        top_k=top_k,
    )

    retrieval_latency_ms = (
        time.perf_counter() - retrieval_start
    ) * 1000

    retrieval = deduplicate_results(retrieval)

    route = retrieval["route"]
    intent = retrieval["intent"]

    with track_rag_run(
        question=query,
        route=route,
        model_name=settings.gemini_model,
    ):
        # --------------------------------------------------
        # 1. Log query intent
        # --------------------------------------------------
        mlflow.log_dict(
            intent,
            "query_intent.json",
        )

        # --------------------------------------------------
        # 2. Extract & log retrieved ad IDs
        # --------------------------------------------------
        sql_ids = [
            item["ad_id"]
            for item in retrieval.get("sql_results", [])
            if isinstance(item, dict) and "ad_id" in item
        ]
        vector_ids = [
            item["ad_id"]
            for item in retrieval.get("vector_results", [])
            if isinstance(item, dict) and "ad_id" in item
        ]
        retrieved_ids = sql_ids + vector_ids

        mlflow.log_dict(
            {
                "retrieved_ad_ids": retrieved_ids,
                "count": len(retrieved_ids),
                "sql_count": len(sql_ids),
                "vector_count": len(vector_ids),
            },
            "retrieval_results.json",
        )

        # --------------------------------------------------
        # 3 & 4. Log retrieval parameters & full RAG config
        # --------------------------------------------------
        mlflow.log_params(
            {
                "top_k": top_k,
                "embedding_model": settings.gemini_embeddings,
                "environment": settings.app_env,
            }
        )

        # Context generation & model answer
        context = build_context(retrieval)

        generation_start = time.perf_counter()

        answer = generate_answer(
            question=query,
            context=context,
        )

        generation_latency_ms = (
            time.perf_counter() - generation_start
        ) * 1000

        retrieved_document_count = len(retrieved_ids)

        # Metrics & Artifact logging
        mlflow.log_metric(
            "retrieval_latency_ms",
            retrieval_latency_ms,
        )

        mlflow.log_metric(
            "generation_latency_ms",
            generation_latency_ms,
        )

        mlflow.log_metric(
            "retrieved_document_count",
            retrieved_document_count,
        )

        mlflow.log_text(
            context,
            "retrieval_context.txt",
        )

        mlflow.log_text(
            answer,
            "generated_answer.txt",
        )

        return {
            "question": query,
            "route": route,
            "intent": intent,
            "answer": answer,
            "context": context,
            "retrieval": retrieval,
        }