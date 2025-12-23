"""
Evaluation Module for Construction RAG System
==============================================

This module implements comprehensive evaluation metrics for RAG systems.
Evaluation is critical for understanding system quality and identifying
areas for improvement.

Why RAG Evaluation is Hard:
--------------------------
1. Multiple components to evaluate (retrieval + generation)
2. No single "correct" answer (many valid responses)
3. Context relevance vs. answer quality
4. Subjective human judgments

Our Evaluation Framework:
-------------------------
1. Retrieval Metrics:
   - Precision@K: % of retrieved docs that are relevant
   - Recall@K: % of relevant docs that are retrieved
   - MRR: Mean Reciprocal Rank of first relevant doc
   - NDCG: Normalized Discounted Cumulative Gain

2. Generation Metrics:
   - Faithfulness: Is the answer grounded in context?
   - Relevance: Does the answer address the question?
   - Completeness: Does the answer cover all aspects?
   - Fluency: Is the answer well-written?

3. End-to-End Metrics:
   - Answer correctness (vs. ground truth)
   - User satisfaction (requires human feedback)
   - Latency and cost

ML Fundamentals Connection:
---------------------------
RAG evaluation parallels ML model evaluation:

Classification analogy:
- True Positive: Relevant doc retrieved
- False Positive: Irrelevant doc retrieved
- False Negative: Relevant doc missed

Precision = TP / (TP + FP) -> Quality of retrieval
Recall = TP / (TP + FN) -> Coverage of retrieval

Just like in ML, there's often a precision-recall tradeoff:
- Higher top_k -> Higher recall, lower precision
- Lower top_k -> Higher precision, lower recall
"""

from typing import List, Dict, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from loguru import logger


@dataclass
class RetrievalEvalResult:
    """Results of retrieval evaluation."""
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    mrr: float
    ndcg: float
    hit_rate: float
    avg_score: float
    latency_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "hit_rate": self.hit_rate,
            "avg_score": self.avg_score,
            "latency_ms": self.latency_ms
        }


@dataclass
class GenerationEvalResult:
    """Results of generation evaluation."""
    faithfulness: float  # 0-1: grounded in context
    relevance: float     # 0-1: answers the question
    completeness: float  # 0-1: covers all aspects
    fluency: float       # 0-1: well-written
    answer_similarity: float  # vs ground truth if available
    hallucination_score: float  # inverse of faithfulness
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "completeness": self.completeness,
            "fluency": self.fluency,
            "answer_similarity": self.answer_similarity,
            "hallucination_score": self.hallucination_score
        }


@dataclass
class EvalExample:
    """Single evaluation example with ground truth."""
    query: str
    relevant_docs: List[str]  # IDs of relevant documents
    ground_truth_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalDataset:
    """Collection of evaluation examples."""
    name: str
    examples: List[EvalExample]
    description: str = ""
    
    def __len__(self) -> int:
        return len(self.examples)


class RetrievalEvaluator:
    """
    Evaluates the retrieval component of the RAG system.
    
    Implements standard IR (Information Retrieval) metrics.
    """
    
    def __init__(self):
        """Initialize retrieval evaluator."""
        logger.info("Retrieval evaluator initialized")
    
    def precision_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int
    ) -> float:
        """
        Calculate Precision@K.
        
        Precision@K = (# relevant in top-K) / K
        
        Example:
        - Retrieved: [A, B, C, D, E]
        - Relevant: {A, C, E}
        - Precision@3 = 2/3 = 0.67 (A and C are relevant in top 3)
        """
        if k == 0:
            return 0.0
        
        top_k = retrieved_ids[:k]
        relevant_in_top_k = len([id for id in top_k if id in relevant_ids])
        
        return relevant_in_top_k / k
    
    def recall_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int
    ) -> float:
        """
        Calculate Recall@K.
        
        Recall@K = (# relevant in top-K) / (total # relevant)
        
        Example:
        - Retrieved: [A, B, C, D, E]
        - Relevant: {A, C, E, X, Y} (5 total relevant)
        - Recall@3 = 2/5 = 0.4 (A and C found in top 3)
        """
        if len(relevant_ids) == 0:
            return 1.0  # All relevant found if none exist
        
        top_k = retrieved_ids[:k]
        relevant_in_top_k = len([id for id in top_k if id in relevant_ids])
        
        return relevant_in_top_k / len(relevant_ids)
    
    def mean_reciprocal_rank(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        MRR = 1 / (position of first relevant result)
        
        Example:
        - Retrieved: [B, A, C] where A is relevant
        - MRR = 1/2 = 0.5 (A is at position 2)
        
        Higher MRR means relevant docs appear earlier.
        """
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / i
        
        return 0.0  # No relevant docs found
    
    def ndcg_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        
        NDCG accounts for position - relevant docs at top matter more.
        
        DCG = sum(rel_i / log2(i + 1)) for i in 1..k
        NDCG = DCG / IDCG (ideal DCG)
        """
        if k == 0 or len(relevant_ids) == 0:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k], 1):
            if doc_id in relevant_ids:
                dcg += 1.0 / np.log2(i + 1)
        
        # Calculate ideal DCG (all relevant docs at top)
        n_relevant = min(k, len(relevant_ids))
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, n_relevant + 1))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate(
        self,
        retrieved_ids: List[str],
        retrieved_scores: List[float],
        relevant_ids: Set[str],
        latency_ms: float = 0.0,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> RetrievalEvalResult:
        """
        Run complete retrieval evaluation.
        
        Args:
            retrieved_ids: List of retrieved document IDs (in order)
            retrieved_scores: Similarity scores for each retrieved doc
            relevant_ids: Set of ground truth relevant document IDs
            latency_ms: Time taken for retrieval
            k_values: K values for precision/recall@K
        
        Returns:
            Complete retrieval evaluation results
        """
        # Calculate metrics at each K
        precision = {k: self.precision_at_k(retrieved_ids, relevant_ids, k) 
                    for k in k_values}
        recall = {k: self.recall_at_k(retrieved_ids, relevant_ids, k)
                 for k in k_values}
        
        # Calculate MRR and NDCG
        mrr = self.mean_reciprocal_rank(retrieved_ids, relevant_ids)
        ndcg = self.ndcg_at_k(retrieved_ids, relevant_ids, max(k_values))
        
        # Hit rate: Did we get at least one relevant doc?
        hit_rate = 1.0 if any(id in relevant_ids for id in retrieved_ids) else 0.0
        
        # Average score of retrieved docs
        avg_score = np.mean(retrieved_scores) if retrieved_scores else 0.0
        
        return RetrievalEvalResult(
            precision_at_k=precision,
            recall_at_k=recall,
            mrr=mrr,
            ndcg=ndcg,
            hit_rate=hit_rate,
            avg_score=float(avg_score),
            latency_ms=latency_ms
        )


class GenerationEvaluator:
    """
    Evaluates the generation component of the RAG system.
    
    Uses both heuristics and optional LLM-based evaluation.
    """
    
    def __init__(self, use_llm_eval: bool = False, llm = None):
        """
        Initialize generation evaluator.
        
        Args:
            use_llm_eval: Use LLM for evaluation (more accurate, costs $)
            llm: LLM instance for evaluation
        """
        self.use_llm_eval = use_llm_eval
        self.llm = llm
        
        logger.info(f"Generation evaluator initialized (LLM eval: {use_llm_eval})")
    
    def _compute_text_overlap(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute word overlap between two texts.
        
        Simple heuristic for faithfulness/relevance.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)  # Jaccard similarity
    
    def _compute_rouge_l(
        self,
        hypothesis: str,
        reference: str
    ) -> float:
        """
        Compute ROUGE-L score (Longest Common Subsequence).
        
        Useful for evaluating text similarity that considers order.
        """
        # Simple implementation without external library
        h_words = hypothesis.lower().split()
        r_words = reference.lower().split()
        
        if not h_words or not r_words:
            return 0.0
        
        # LCS using dynamic programming
        m, n = len(h_words), len(r_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if h_words[i-1] == r_words[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        
        # F1-style score
        precision = lcs_length / m if m > 0 else 0
        recall = lcs_length / n if n > 0 else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def faithfulness_score(
        self,
        answer: str,
        context: str
    ) -> float:
        """
        Evaluate faithfulness: Is the answer grounded in context?
        
        A faithful answer only contains information from the context.
        """
        # Extract potential claims from answer (sentences)
        import re
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return 1.0
        
        # Check each sentence against context
        grounded_count = 0
        for sentence in sentences:
            overlap = self._compute_text_overlap(sentence, context)
            if overlap > 0.2:  # Threshold for "grounded"
                grounded_count += 1
        
        return grounded_count / len(sentences)
    
    def relevance_score(
        self,
        answer: str,
        question: str
    ) -> float:
        """
        Evaluate relevance: Does the answer address the question?
        
        Check if answer content relates to question topic.
        """
        # Extract key terms from question
        question_words = set(question.lower().split())
        stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 'do', 'does', 'are', 'for', 'in', 'to', 'of'}
        key_words = question_words - stop_words
        
        if not key_words:
            return 1.0
        
        answer_lower = answer.lower()
        found_keywords = sum(1 for word in key_words if word in answer_lower)
        
        return found_keywords / len(key_words)
    
    def completeness_score(
        self,
        answer: str,
        question: str
    ) -> float:
        """
        Evaluate completeness: Does the answer cover all aspects?
        
        Heuristic: Longer, structured answers are more complete.
        """
        # Check for common completeness indicators
        indicators = {
            'structured': bool(re.search(r'\d\.|•|-\s', answer)),  # Lists
            'multiple_points': answer.count('.') >= 3,  # Multiple sentences
            'adequate_length': len(answer.split()) >= 20,  # Not too short
            'has_examples': 'example' in answer.lower() or 'e.g.' in answer.lower(),
        }
        
        score = sum(indicators.values()) / len(indicators)
        
        # If answer says "don't know", lower score
        if 'don\'t have information' in answer.lower() or 'cannot find' in answer.lower():
            score *= 0.5
        
        return score
    
    def fluency_score(self, answer: str) -> float:
        """
        Evaluate fluency: Is the answer well-written?
        
        Heuristic checks for basic writing quality.
        """
        # Check for sentence structure
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Score components
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        has_good_length = 5 <= avg_sentence_length <= 25
        
        # Check for capitalization
        properly_capitalized = sum(1 for s in sentences if s[0].isupper()) / len(sentences)
        
        # Check for basic coherence (no obvious issues)
        no_obvious_issues = not bool(re.search(r'(\w+)\s+\1\s+\1', answer))  # No triple word repetition
        
        score = (
            (0.4 if has_good_length else 0.2) +
            (0.3 * properly_capitalized) +
            (0.3 if no_obvious_issues else 0.0)
        )
        
        return min(score, 1.0)
    
    def answer_similarity(
        self,
        generated_answer: str,
        ground_truth: str
    ) -> float:
        """
        Compare generated answer to ground truth (if available).
        """
        if not ground_truth:
            return -1.0  # Not applicable
        
        return self._compute_rouge_l(generated_answer, ground_truth)
    
    def evaluate(
        self,
        answer: str,
        question: str,
        context: str,
        ground_truth: Optional[str] = None
    ) -> GenerationEvalResult:
        """
        Run complete generation evaluation.
        
        Args:
            answer: Generated answer
            question: User question
            context: Retrieved context used
            ground_truth: Expected correct answer (optional)
        
        Returns:
            Complete generation evaluation results
        """
        faithfulness = self.faithfulness_score(answer, context)
        relevance = self.relevance_score(answer, question)
        completeness = self.completeness_score(answer, question)
        fluency = self.fluency_score(answer)
        similarity = self.answer_similarity(answer, ground_truth) if ground_truth else 0.0
        
        return GenerationEvalResult(
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
            fluency=fluency,
            answer_similarity=similarity,
            hallucination_score=1.0 - faithfulness
        )


class RAGEvaluator:
    """
    End-to-end RAG system evaluator.
    
    Combines retrieval and generation evaluation
    for comprehensive system assessment.
    """
    
    def __init__(self):
        """Initialize RAG evaluator."""
        self.retrieval_evaluator = RetrievalEvaluator()
        self.generation_evaluator = GenerationEvaluator()
        
        logger.info("RAG evaluator initialized")
    
    def evaluate_single(
        self,
        query: str,
        retrieved_ids: List[str],
        retrieved_scores: List[float],
        retrieved_content: List[str],
        generated_answer: str,
        relevant_ids: Set[str],
        ground_truth_answer: Optional[str] = None,
        latency_ms: float = 0.0
    ) -> Dict[str, Any]:
        """
        Evaluate a single RAG query.
        
        Returns combined retrieval and generation metrics.
        """
        # Evaluate retrieval
        retrieval_result = self.retrieval_evaluator.evaluate(
            retrieved_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            relevant_ids=relevant_ids,
            latency_ms=latency_ms
        )
        
        # Build context from retrieved content
        context = "\n\n".join(retrieved_content)
        
        # Evaluate generation
        generation_result = self.generation_evaluator.evaluate(
            answer=generated_answer,
            question=query,
            context=context,
            ground_truth=ground_truth_answer
        )
        
        return {
            "query": query,
            "retrieval": retrieval_result.to_dict(),
            "generation": generation_result.to_dict()
        }
    
    def evaluate_dataset(
        self,
        dataset: EvalDataset,
        rag_function: callable
    ) -> Dict[str, Any]:
        """
        Evaluate RAG system on an evaluation dataset.
        
        Args:
            dataset: EvalDataset with examples
            rag_function: Function(query) -> (retrieved_ids, scores, content, answer)
        
        Returns:
            Aggregated evaluation results
        """
        all_results = []
        
        for example in dataset.examples:
            # Run RAG
            retrieved_ids, scores, content, answer = rag_function(example.query)
            
            # Evaluate
            result = self.evaluate_single(
                query=example.query,
                retrieved_ids=retrieved_ids,
                retrieved_scores=scores,
                retrieved_content=content,
                generated_answer=answer,
                relevant_ids=set(example.relevant_docs),
                ground_truth_answer=example.ground_truth_answer
            )
            
            all_results.append(result)
        
        # Aggregate results
        return self._aggregate_results(all_results)
    
    def _aggregate_results(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate results across multiple examples."""
        if not results:
            return {}
        
        # Collect metrics
        retrieval_metrics = defaultdict(list)
        generation_metrics = defaultdict(list)
        
        for result in results:
            for k, v in result["retrieval"].items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        retrieval_metrics[f"{k}_{sub_k}"].append(sub_v)
                else:
                    retrieval_metrics[k].append(v)
            
            for k, v in result["generation"].items():
                generation_metrics[k].append(v)
        
        # Calculate averages
        avg_retrieval = {k: np.mean(v) for k, v in retrieval_metrics.items()}
        avg_generation = {k: np.mean(v) for k, v in generation_metrics.items()
                         if all(vv >= 0 for vv in v)}  # Skip -1 (N/A)
        
        return {
            "num_examples": len(results),
            "retrieval": avg_retrieval,
            "generation": avg_generation,
            "individual_results": results
        }


def create_sample_eval_dataset() -> EvalDataset:
    """Create a sample evaluation dataset for construction RAG."""
    examples = [
        EvalExample(
            query="What PPE is required on construction sites?",
            relevant_docs=["construction_safety_manual.txt"],
            ground_truth_answer="Required PPE includes hard hats (ANSI Z89.1 certified), "
                               "safety glasses (ANSI Z87.1), high-visibility vests, "
                               "and steel-toed boots."
        ),
        EvalExample(
            query="When is fall protection required?",
            relevant_docs=["construction_safety_manual.txt"],
            ground_truth_answer="Fall protection is required when working at heights "
                               "of 6 feet or more above lower levels."
        ),
        EvalExample(
            query="What is the contract value for electrical work?",
            relevant_docs=["subcontractor_agreement.txt"],
            ground_truth_answer="The contract value for electrical work is "
                               "$4,852,000.00."
        ),
        EvalExample(
            query="What inspection deficiencies require immediate correction?",
            relevant_docs=["site_inspection_report.txt"],
            ground_truth_answer="Critical deficiencies requiring immediate correction "
                               "include missing floor opening covers on Floor 14, "
                               "combustible storage in electrical room, and "
                               "damaged extension cords."
        ),
    ]
    
    return EvalDataset(
        name="construction_rag_eval",
        examples=examples,
        description="Sample evaluation dataset for construction RAG"
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Evaluation Module Demo")
    print("=" * 60)
    
    # Demo retrieval evaluation
    print("\n📊 Retrieval Evaluation Demo\n")
    
    evaluator = RetrievalEvaluator()
    
    # Simulated retrieval results
    retrieved = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
    scores = [0.95, 0.85, 0.75, 0.65, 0.55]
    relevant = {"doc_1", "doc_3", "doc_7"}  # doc_7 not retrieved
    
    result = evaluator.evaluate(retrieved, scores, relevant)
    
    print(f"  Precision@1: {result.precision_at_k[1]:.2f}")
    print(f"  Precision@3: {result.precision_at_k[3]:.2f}")
    print(f"  Recall@3: {result.recall_at_k[3]:.2f}")
    print(f"  MRR: {result.mrr:.2f}")
    print(f"  NDCG: {result.ndcg:.2f}")
    print(f"  Hit Rate: {result.hit_rate:.2f}")
    
    # Demo generation evaluation
    print("\n📊 Generation Evaluation Demo\n")
    
    gen_evaluator = GenerationEvaluator()
    
    answer = "Fall protection is required when working at heights of 6 feet or more."
    question = "When is fall protection required?"
    context = "Fall protection is mandatory when working at heights of 6 feet or more above lower levels. Workers must use safety harnesses and guardrails."
    ground_truth = "Fall protection is required above 6 feet."
    
    gen_result = gen_evaluator.evaluate(answer, question, context, ground_truth)
    
    print(f"  Faithfulness: {gen_result.faithfulness:.2f}")
    print(f"  Relevance: {gen_result.relevance:.2f}")
    print(f"  Completeness: {gen_result.completeness:.2f}")
    print(f"  Fluency: {gen_result.fluency:.2f}")
    print(f"  Answer Similarity: {gen_result.answer_similarity:.2f}")
    print(f"  Hallucination Score: {gen_result.hallucination_score:.2f}")
