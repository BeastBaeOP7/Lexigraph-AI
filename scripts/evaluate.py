import time
import json
import csv
from pathlib import Path
from typing import List, Dict, Any

from app.config.settings import settings
from app.config.logging import logger
from app.graph.graph import app_graph
from app.evaluation.testset import GoldenDatasetLoader
from app.evaluation.judge_eval import LLMJudgeEvaluator
from app.retrieval.hybrid.hybrid_search import HybridRetrievalEngine

def run_evaluation() -> None:
    """Runs the evaluation pipeline against the golden dataset and writes reports."""
    logger.info("Initializing evaluation pipeline...")
    
    golden_path = Path("data/golden_dataset.json")
    if not golden_path.exists():
        logger.error(f"Golden dataset not found at: {golden_path}")
        return
        
    dataset = GoldenDatasetLoader.load_dataset(golden_path)
    logger.info(f"Loaded {len(dataset)} evaluation samples.")
    
    evaluator = LLMJudgeEvaluator()
    engine = HybridRetrievalEngine()
    
    results: List[Dict[str, Any]] = []
    
    # Track overall aggregation metrics
    total_samples = len(dataset)
    sum_faithfulness = 0.0
    sum_relevancy = 0.0
    sum_precision = 0.0
    sum_recall = 0.0
    
    sum_retrieval_lat = 0.0
    sum_rerank_lat = 0.0
    sum_gen_lat = 0.0
    sum_total_lat = 0.0
    
    total_citations = 0
    total_missing_citations = 0
    total_failures = 0
    
    for idx, sample in enumerate(dataset):
        question = sample["question"]
        expected_answer = sample["expected_answer"]
        expected_doc = sample["expected_document"]
        
        logger.info(f"[{idx+1}/{total_samples}] Evaluating: '{question[:40]}...'")
        
        # 1. Measure Latency Components
        try:
            # Retrieval Latency
            start_ret = time.time()
            bm25_res = engine.bm25_searcher.search(question, top_k=settings.bm25_k)
            vector_res = engine.vector_searcher.search(question, top_k=settings.vector_k)
            fused_res = engine.reciprocal_rank_fusion(bm25_res, vector_res)
            ret_lat = time.time() - start_ret
            
            # Reranking Latency
            start_rerank = time.time()
            candidates = [c[0] for c in fused_res]
            reranked = engine.reranker.rerank(question, candidates)
            rerank_lat = time.time() - start_rerank
            
            # Full Pipeline (Total) Latency
            initial_state = {
                "user_query": question,
                "retrieved_chunks": [],
                "validated_chunks": [],
                "generated_answer": "",
                "citations": [],
                "confidence_score": 0.0,
                "errors": [],
                "metadata": {}
            }
            start_total = time.time()
            state_output = app_graph.invoke(initial_state)
            total_lat = time.time() - start_total
            
            # Estimate Generation Latency
            gen_lat = max(0.0, total_lat - ret_lat - rerank_lat)
            
        except Exception as e:
            logger.error(f"Pipeline execution failed for sample {idx+1}: {e}")
            total_failures += 1
            continue

        # Extract answer and context text
        ans = state_output.get("generated_answer", "")
        # Combined context text from validated chunks
        validated_chunks = state_output.get("validated_chunks", [])
        context_text = "\n\n".join(c.text for c in validated_chunks)
        chunk_texts = [c.text for c in validated_chunks]
        
        # 2. Ragas LLM-Judge Evaluation Metrics
        logger.info("Computing metrics...")
        faithfulness = evaluator.evaluate_faithfulness(ans, context_text)
        relevancy = evaluator.evaluate_answer_relevancy(question, ans)
        precision = evaluator.evaluate_context_precision(question, chunk_texts)
        recall = evaluator.evaluate_context_recall(expected_answer, context_text)
        
        # Aggregate scores
        sum_faithfulness += faithfulness
        sum_relevancy += relevancy
        sum_precision += precision
        sum_recall += recall
        
        sum_retrieval_lat += ret_lat
        sum_rerank_lat += rerank_lat
        sum_gen_lat += gen_lat
        sum_total_lat += total_lat
        
        # Citations
        citations_list = state_output.get("citations", [])
        total_citations += len(citations_list)
        
        # Errors (e.g. removed unverified citations)
        errors_list = state_output.get("metadata", {}).get("errors", [])
        removed_cit_count = sum(1 for e in errors_list if "Removed unverified citation" in e)
        total_missing_citations += removed_cit_count

        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "answer": ans,
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_precision": precision,
            "context_recall": recall,
            "retrieval_latency": ret_lat,
            "rerank_latency": rerank_lat,
            "generation_latency": gen_lat,
            "total_latency": total_lat,
            "citations_count": len(citations_list),
            "missing_citations_count": removed_cit_count,
            "confidence_score": state_output.get("confidence_score", 0.0)
        })

    # Overall stats
    denom = len(results) or 1
    averages = {
        "faithfulness": sum_faithfulness / denom,
        "answer_relevancy": sum_relevancy / denom,
        "context_precision": sum_precision / denom,
        "context_recall": sum_recall / denom,
        "retrieval_latency": sum_retrieval_lat / denom,
        "rerank_latency": sum_rerank_lat / denom,
        "generation_latency": sum_gen_lat / denom,
        "total_latency": sum_total_lat / denom,
        "average_confidence": sum(r["confidence_score"] for r in results) / denom
    }
    
    thresholds = {
        "faithfulness": settings.faithfulness_threshold,
        "answer_relevancy": settings.answer_relevancy_threshold,
        "latency": settings.latency_threshold
    }

    report_dir = Path("data/evaluations")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. JSON Report
    json_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "thresholds": thresholds,
        "overall_averages": averages,
        "statistics": {
            "total_samples": total_samples,
            "completed_samples": len(results),
            "failures": total_failures,
            "total_citations": total_citations,
            "total_missing_citations": total_missing_citations
        },
        "individual_results": results
    }
    
    with open(report_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)
        
    # 2. CSV Report
    with open(report_dir / "report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Question", "Expected Answer", "Generated Answer",
            "Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall",
            "Retrieval Latency", "Rerank Latency", "Generation Latency", "Total Latency",
            "Citations Count", "Missing Citations Count", "Confidence Score"
        ])
        for r in results:
            writer.writerow([
                r["question"], r["expected_answer"], r["answer"],
                f"{r['faithfulness']:.3f}", f"{r['answer_relevancy']:.3f}", 
                f"{r['context_precision']:.3f}", f"{r['context_recall']:.3f}",
                f"{r['retrieval_latency']:.4f}", f"{r['rerank_latency']:.4f}", 
                f"{r['generation_latency']:.4f}", f"{r['total_latency']:.4f}",
                r["citations_count"], r["missing_citations_count"], 
                f"{r['confidence_score']:.3f}"
            ])

    # 3. HTML Report
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Platform Evaluation Report</title>
    <style>
        body {{ font-family: 'Inter', Arial, sans-serif; background: #f8fafc; color: #1e293b; padding: 2rem; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); padding: 2rem; }}
        h1 {{ background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1.5rem; }}
        .card {{ background: #f1f5f9; border-radius: 8px; padding: 1rem; border-left: 4px solid #2a5298; }}
        .card-label {{ font-size: 0.8rem; text-transform: uppercase; color: #64748b; font-weight: bold; }}
        .card-value {{ font-size: 1.5rem; font-weight: bold; margin-top: 0.25rem; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 2rem; font-size: 0.9rem; }}
        th, td {{ padding: 0.75rem; border: 1px solid #e2e8f0; text-align: left; }}
        th {{ background: #f8fafc; font-weight: 600; }}
        .pass {{ color: green; font-weight: bold; }}
        .fail {{ color: red; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Legal Contract Platform Evaluation Report</h1>
        <p>Generated on: {json_report['timestamp']}</p>
        
        <h2>Overall Averages</h2>
        <div class="grid">
            <div class="card">
                <div class="card-label">Faithfulness</div>
                <div class="card-value">{averages['faithfulness']:.3f}</div>
            </div>
            <div class="card">
                <div class="card-label">Answer Relevancy</div>
                <div class="card-value">{averages['answer_relevancy']:.3f}</div>
            </div>
            <div class="card">
                <div class="card-label">Context Recall</div>
                <div class="card-value">{averages['context_recall']:.3f}</div>
            </div>
            <div class="card">
                <div class="card-label">Total Latency</div>
                <div class="card-value">{averages['total_latency']:.3f}s</div>
            </div>
        </div>

        <h2>Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Question</th>
                    <th>Faithfulness</th>
                    <th>Relevancy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>Total Latency</th>
                    <th>Citations</th>
                </tr>
            </thead>
            <tbody>
    """
    for r in results:
        html_content += f"""
                <tr>
                    <td>{r['question']}</td>
                    <td class="{'pass' if r['faithfulness']>=settings.faithfulness_threshold else 'fail'}">{r['faithfulness']:.3f}</td>
                    <td class="{'pass' if r['answer_relevancy']>=settings.answer_relevancy_threshold else 'fail'}">{r['answer_relevancy']:.3f}</td>
                    <td>{r['context_precision']:.3f}</td>
                    <td>{r['context_recall']:.3f}</td>
                    <td class="{'pass' if r['total_latency']<=settings.latency_threshold else 'fail'}">{r['total_latency']:.3f}s</td>
                    <td>{r['citations_count']}</td>
                </tr>
        """
    html_content += """
            </tbody>
        </table>
    </div>
</body>
</html>
    """
    
    with open(report_dir / "report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("Evaluation complete! Reports generated in data/evaluations/.")

    # CI threshold check
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", action="store_true", help="Fail if thresholds are not met")
    args, _ = parser.parse_known_args()
    
    if args.ci:
        if averages["faithfulness"] < settings.faithfulness_threshold:
            logger.error(f"CI failed: Faithfulness {averages['faithfulness']:.2f} < threshold {settings.faithfulness_threshold}")
            sys.exit(1)
        if averages["answer_relevancy"] < settings.answer_relevancy_threshold:
            logger.error(f"CI failed: Answer Relevancy {averages['answer_relevancy']:.2f} < threshold {settings.answer_relevancy_threshold}")
            sys.exit(1)
        logger.info("CI threshold verification passed!")

if __name__ == "__main__":
    run_evaluation()
