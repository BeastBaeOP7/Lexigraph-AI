import json
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from app.nodes.generate import get_openai_client
from app.indexing.indexer import Indexer

class LLMJudgeEvaluator:
    """LLM-judge evaluator calculating Faithfulness, Relevancy, Precision, and Recall."""
    
    def __init__(self, indexer: Indexer = None) -> None:
        self.indexer = indexer or Indexer()

    def _call_llm_json(self, system_prompt: str, user_prompt: str) -> Any:
        """Calls GPT and parses JSON output block."""
        client = get_openai_client()
        try:
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            content = res.choices[0].message.content or ""
            clean_content = re.sub(r'^```json\s*|\s*```$', '', content.strip(), flags=re.MULTILINE)
            return json.loads(clean_content)
        except Exception as e:
            return None

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Faithfulness = (statements supported by context) / (total statements in answer).
        """
        if not answer.strip() or not context.strip():
            return 0.0
            
        system_prompt = (
            "You are a factual accuracy judge. Extract all standalone statements/claims "
            "made in the generated answer. Then, for each statement, check if it is "
            "directly supported or can be logically inferred from the retrieved context.\n"
            "Respond ONLY with a JSON array of objects, containing 'statement' (string) "
            "and 'supported' (boolean). Format:\n"
            '[{"statement": "...", "supported": true}]'
        )
        user_prompt = f"Context:\n{context}\n\nGenerated Answer:\n{answer}"
        
        result = self._call_llm_json(system_prompt, user_prompt)
        if not result or not isinstance(result, list):
            return 0.5

        supported_count = sum(1 for item in result if item.get("supported") is True)
        return float(supported_count / len(result)) if result else 1.0

    def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Answer Relevancy = average cosine similarity between original question and questions
        generated from the answer.
        """
        if not question.strip() or not answer.strip():
            return 0.0
            
        system_prompt = (
            "Based on the provided answer, write three distinct, short questions that this "
            "answer directly addresses. Respond ONLY with a JSON array of strings. Format:\n"
            '["question 1", "question 2", "question 3"]'
        )
        user_prompt = f"Answer:\n{answer}"
        
        result = self._call_llm_json(system_prompt, user_prompt)
        if not result or not isinstance(result, list):
            return 0.5

        try:
            q_emb = self.indexer.embedding_model.encode(question, normalize_embeddings=True)
            gen_embs = self.indexer.embedding_model.encode(result, normalize_embeddings=True)
            
            similarities = []
            for g_emb in gen_embs:
                sim = float(np.dot(q_emb, g_emb))
                similarities.append(sim)
            return float(np.mean(similarities)) if similarities else 0.5
        except Exception:
            return 0.5

    def evaluate_context_recall(self, expected_answer: str, context: str) -> float:
        """
        Context Recall = (ground truth sentences found in context) / (total ground truth sentences).
        """
        if not expected_answer.strip() or not context.strip():
            return 0.0

        system_prompt = (
            "Analyze the expected answer sentence-by-sentence. For each sentence, determine "
            "if it is present or directly supported by the retrieved context.\n"
            "Respond ONLY with a JSON array of objects containing 'sentence' (string) "
            "and 'supported' (boolean). Format:\n"
            '[{"sentence": "...", "supported": true}]'
        )
        user_prompt = f"Context:\n{context}\n\nExpected Answer:\n{expected_answer}"
        
        result = self._call_llm_json(system_prompt, user_prompt)
        if not result or not isinstance(result, list):
            return 0.5

        supported_count = sum(1 for item in result if item.get("supported") is True)
        return float(supported_count / len(result)) if result else 1.0

    def evaluate_context_precision(self, question: str, retrieved_chunks: List[str]) -> float:
        """
        Context Precision = mean Precision@k for all relevant chunks.
        """
        if not question.strip() or not retrieved_chunks:
            return 0.0

        system_prompt = (
            "For each retrieved chunk, check if it contains relevant information to answer "
            "the user's question. Respond ONLY with a JSON array of booleans corresponding "
            "to each chunk in order. Format:\n"
            "[true, false, true]"
        )
        chunks_str = "\n\n".join(f"Chunk {i+1}:\n{text}" for i, text in enumerate(retrieved_chunks))
        user_prompt = f"Question:\n{question}\n\nChunks:\n{chunks_str}"

        relevances = self._call_llm_json(system_prompt, user_prompt)
        if not relevances or not isinstance(relevances, list) or len(relevances) != len(retrieved_chunks):
            relevances = [True] * len(retrieved_chunks)

        precision_scores = []
        relevant_so_far = 0
        for idx, is_rel in enumerate(relevances):
            if is_rel:
                relevant_so_far += 1
                precision_at_k = relevant_so_far / (idx + 1)
                precision_scores.append(precision_at_k)

        if not precision_scores:
            return 0.0
            
        return float(np.mean(precision_scores))
