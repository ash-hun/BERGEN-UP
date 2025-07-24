from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union
import uuid

class BaseRetrievalEvaluation(ABC):
    """
    Base class for retrieval evaluation frameworks.
    
    Input format:
    {
        'question': Dict[uuid, str],
        'context_gt_map': Dict[uuid, int], 
        'context_gt': Dict[int, str],
        'retrieved': Dict[uuid, List[str]]
    }
    
    Output format:
    {
        'top_k': Dict[metric, score],
        ...
    }
    """
    
    def __init__(self):
        self.supported_metrics = [
            'precision', 'recall', 'f1', 'mrr', 'ndcg', 'hit_rate'
        ]
    
    @abstractmethod
    def evaluate(self, data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate retrieval performance using the given data.
        
        Args:
            data: Dictionary containing questions, ground truth, and retrieved results
            
        Returns:
            Dictionary with evaluation results for different top-k values
        """
        pass
    
    def _validate_input(self, data: Dict[str, Any]) -> bool:
        """
        Validate input data format.
        
        Args:
            data: Input data dictionary
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        required_keys = ['question', 'context_gt_map', 'context_gt', 'retrieved']
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        # Validate question format
        if not isinstance(data['question'], dict):
            raise ValueError("'question' must be a dictionary")
        
        # Validate context_gt_map format
        if not isinstance(data['context_gt_map'], dict):
            raise ValueError("'context_gt_map' must be a dictionary")
        
        # Validate context_gt format
        if not isinstance(data['context_gt'], dict):
            raise ValueError("'context_gt' must be a dictionary")
        
        # Validate retrieved format
        if not isinstance(data['retrieved'], dict):
            raise ValueError("'retrieved' must be a dictionary")
        
        # Check if all question UUIDs have corresponding retrieved results
        for question_uuid in data['question'].keys():
            if question_uuid not in data['retrieved']:
                raise ValueError(f"No retrieved results for question UUID: {question_uuid}")
        
        return True
    
    def _calculate_precision_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate precision@k.
        
        Args:
            retrieved: List of retrieved documents
            relevant: List of relevant documents
            k: Number of top documents to consider
            
        Returns:
            Precision@k score
        """
        if k == 0:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_at_k) & set(relevant))
        
        return relevant_retrieved / min(k, len(retrieved_at_k))
    
    def _calculate_recall_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate recall@k.
        
        Args:
            retrieved: List of retrieved documents
            relevant: List of relevant documents
            k: Number of top documents to consider
            
        Returns:
            Recall@k score
        """
        if len(relevant) == 0:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_at_k) & set(relevant))
        
        return relevant_retrieved / len(relevant)
    
    def _calculate_f1_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate F1@k.
        
        Args:
            retrieved: List of retrieved documents
            relevant: List of relevant documents
            k: Number of top documents to consider
            
        Returns:
            F1@k score
        """
        precision = self._calculate_precision_at_k(retrieved, relevant, k)
        recall = self._calculate_recall_at_k(retrieved, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_mrr(self, retrieved: List[str], relevant: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            retrieved: List of retrieved documents
            relevant: List of relevant documents
            
        Returns:
            MRR score
        """
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def _calculate_mrr_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate Mean Reciprocal Rank at k (MRR@k).
        
        Args:
            retrieved: List of retrieved documents
            relevant: List of relevant documents
            k: Number of top documents to consider
            
        Returns:
            MRR@k score
        """
        # Only consider top-k documents
        retrieved_at_k = retrieved[:k]
        
        for i, doc in enumerate(retrieved_at_k):
            if doc in relevant:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def _calculate_hit_rate_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate hit rate@k.
        
        Args:
            retrieved: List of retrieved documents
            relevant: List of relevant documents
            k: Number of top documents to consider
            
        Returns:
            Hit rate@k score (1.0 if at least one relevant doc in top-k, 0.0 otherwise)
        """
        retrieved_at_k = retrieved[:k]
        return 1.0 if any(doc in relevant for doc in retrieved_at_k) else 0.0
    
    def _calculate_ndcg_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate NDCG@k (simplified version - treats all relevant docs as equally relevant).
        
        Args:
            retrieved: List of retrieved documents
            relevant: List of relevant documents
            k: Number of top documents to consider
            
        Returns:
            NDCG@k score
        """
        import math
        
        if len(relevant) == 0:
            return 0.0
        
        # DCG calculation
        dcg = 0.0
        for i, doc in enumerate(retrieved[:k]):
            if doc in relevant:
                dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0
        
        # IDCG calculation (ideal DCG)
        idcg = 0.0
        for i in range(min(k, len(relevant))):
            idcg += 1.0 / math.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0