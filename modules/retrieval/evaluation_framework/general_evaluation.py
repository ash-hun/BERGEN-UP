from typing import Dict, List, Any
from .base_evaluation import BaseRetrievalEvaluation

class GeneralRetrievalEvaluation(BaseRetrievalEvaluation):
    """
    General retrieval evaluation implementation.
    
    Evaluates retrieval performance using standard metrics:
    - Precision@k
    - Recall@k
    - F1@k
    - MRR (Mean Reciprocal Rank)
    - NDCG@k (Normalized Discounted Cumulative Gain)
    - Hit Rate@k
    """
    
    def __init__(self, top_k_values: List[int] = None):
        super().__init__()
        self.top_k_values = top_k_values or [1, 3, 5, 10, 20]
    
    def evaluate(self, data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate retrieval performance.
        
        Args:
            data: Input data dictionary with required keys:
                - question: Dict[uuid, str]
                - context_gt_map: Dict[uuid, int]
                - context_gt: Dict[int, str]
                - retrieved: Dict[uuid, List[str]]
        
        Returns:
            Dictionary with evaluation results:
            {
                'top_1': {'precision': 0.8, 'recall': 0.2, ...},
                'top_3': {'precision': 0.7, 'recall': 0.4, ...},
                ...
                'mrr': {'mrr': 0.65}
            }
        """
        # Validate input
        self._validate_input(data)
        
        # Initialize results dictionary
        results = {}
        
        # Calculate metrics for each top-k value
        for k in self.top_k_values:
            results[f'top_{k}'] = self._calculate_metrics_at_k(data, k)
        
        # Calculate MRR (independent of k)
        results['mrr'] = {'mrr': self._calculate_average_mrr(data)}
        
        return results
    
    def _calculate_metrics_at_k(self, data: Dict[str, Any], k: int) -> Dict[str, float]:
        """
        Calculate all metrics at top-k.
        
        Args:
            data: Input data dictionary
            k: Number of top documents to consider
            
        Returns:
            Dictionary with metric scores
        """
        metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'ndcg': 0.0,
            'hit_rate': 0.0
        }
        
        question_count = len(data['question'])
        
        if question_count == 0:
            return metrics
        
        # Calculate metrics for each question
        for question_uuid in data['question'].keys():
            # Get retrieved documents for this question
            retrieved = data['retrieved'][question_uuid]
            
            # Get relevant documents (ground truth)
            relevant = self._get_relevant_documents(question_uuid, data)
            
            # Calculate metrics
            metrics['precision'] += self._calculate_precision_at_k(retrieved, relevant, k)
            metrics['recall'] += self._calculate_recall_at_k(retrieved, relevant, k)
            metrics['f1'] += self._calculate_f1_at_k(retrieved, relevant, k)
            metrics['ndcg'] += self._calculate_ndcg_at_k(retrieved, relevant, k)
            metrics['hit_rate'] += self._calculate_hit_rate_at_k(retrieved, relevant, k)
        
        # Average the metrics
        for metric in metrics:
            metrics[metric] /= question_count
        
        return metrics
    
    def _calculate_average_mrr(self, data: Dict[str, Any]) -> float:
        """
        Calculate average MRR across all questions.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Average MRR score
        """
        total_mrr = 0.0
        question_count = len(data['question'])
        
        if question_count == 0:
            return 0.0
        
        for question_uuid in data['question'].keys():
            retrieved = data['retrieved'][question_uuid]
            relevant = self._get_relevant_documents(question_uuid, data)
            
            total_mrr += self._calculate_mrr(retrieved, relevant)
        
        return total_mrr / question_count
    
    def _get_relevant_documents(self, question_uuid: str, data: Dict[str, Any]) -> List[str]:
        """
        Get relevant documents for a given question UUID.
        
        Args:
            question_uuid: UUID of the question
            data: Input data dictionary
            
        Returns:
            List of relevant document strings
        """
        # Get the ground truth context ID for this question
        if question_uuid not in data['context_gt_map']:
            return []
        
        context_gt_id = data['context_gt_map'][question_uuid]
        
        # Get the ground truth context
        if context_gt_id not in data['context_gt']:
            return []
        
        context_gt = data['context_gt'][context_gt_id]
        
        # Return as list (assuming single ground truth per question)
        return [context_gt]
    
    def get_summary_report(self, results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Generate a summary report of the evaluation results.
        
        Args:
            results: Results dictionary from evaluate()
            
        Returns:
            Summary report dictionary
        """
        summary = {
            'total_metrics': len(self.supported_metrics),
            'top_k_values': self.top_k_values,
            'best_precision': {'value': 0.0, 'at_k': 0},
            'best_recall': {'value': 0.0, 'at_k': 0},
            'best_f1': {'value': 0.0, 'at_k': 0},
            'best_ndcg': {'value': 0.0, 'at_k': 0},
            'best_hit_rate': {'value': 0.0, 'at_k': 0},
            'mrr': results.get('mrr', {}).get('mrr', 0.0)
        }
        
        # Find best scores for each metric
        for k in self.top_k_values:
            top_k_key = f'top_{k}'
            if top_k_key in results:
                metrics = results[top_k_key]
                
                for metric in ['precision', 'recall', 'f1', 'ndcg', 'hit_rate']:
                    if metric in metrics:
                        if metrics[metric] > summary[f'best_{metric}']['value']:
                            summary[f'best_{metric}']['value'] = metrics[metric]
                            summary[f'best_{metric}']['at_k'] = k
        
        return summary