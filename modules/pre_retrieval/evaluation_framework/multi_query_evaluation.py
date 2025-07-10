from typing import Dict, Any, List
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm
from .base_evaluation import BasePreRetrievalEvaluation


class EvaluationResult(BaseModel):
    """Pydantic model for evaluation results"""
    score: float = Field(ge=0.0, le=1.0, description="Evaluation score between 0 and 1")
    reasoning: str = Field(description="Brief explanation of the evaluation")


class MultiQueryEvaluation(BasePreRetrievalEvaluation):
    """
    Evaluation for Multi-Query generation strategy using LLM-as-a-judge.
    
    Evaluates the quality of generated multiple queries from an original query.
    """
    
    def __init__(self, openai_api_key: str):
        super().__init__(openai_api_key)
        self.validate_api_key()
        self.client = instructor.from_openai(OpenAI(api_key=openai_api_key))
    
    def evaluate(self, data: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluate multi-query generation quality.
        
        Args:
            data: Dictionary containing original_query and multi_queries
            verbose: Whether to show detailed progress
            
        Returns:
            Dictionary with evaluation results
        """
        results = {}
        uuids = list(data['original_query'].keys())
        
        with tqdm(total=len(uuids), desc="Evaluating multi-query generation", disable=not verbose) as pbar:
            for uuid in uuids:
                original_query = data['original_query'][uuid]
                multi_queries = data['multi_queries'][uuid]
                
                # Evaluate diversity
                diversity_score = self.evaluate_diversity(multi_queries)
                
                # Evaluate coverage
                coverage_score = self.evaluate_coverage(original_query, multi_queries)
                
                # Evaluate relevance
                relevance_score = self.evaluate_relevance(original_query, multi_queries)
                
                results[uuid] = {
                    'diversity': diversity_score,
                    'coverage': coverage_score,
                    'relevance': relevance_score
                }
                
                if verbose:
                    pbar.set_postfix({
                        'UUID': uuid,
                        'D': f"{diversity_score:.3f}",
                        'C': f"{coverage_score:.3f}",
                        'R': f"{relevance_score:.3f}"
                    })
                
                pbar.update(1)
        
        return results
    
    def evaluate_diversity(self, multi_queries: List[str]) -> float:
        """
        Evaluate how diverse the generated queries are from each other.
        
        Args:
            multi_queries: List of generated queries
            
        Returns:
            Diversity score (0-1)
        """
        queries_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(multi_queries)])
        
        prompt = f"""You are evaluating the diversity of multiple generated queries.

Generated Queries:
{queries_text}

Evaluation Criteria:
1. Do the queries explore different aspects or perspectives?
2. Is there minimal redundancy between queries?
3. Does each query add unique value?

Please evaluate the diversity of these queries on a scale from 0 to 1, where:
- 0: All queries are nearly identical or highly redundant
- 0.5: Some variation exists but with significant overlap
- 1: Each query explores a unique aspect with minimal overlap
"""
        
        try:
            result = self.client.chat.completions.create(
                model="gpt-4",
                response_model=EvaluationResult,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for query diversity."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            return result.score
            
        except Exception as e:
            self.console.log(f"[red]Error in diversity evaluation: {str(e)}[/red]")
            return 0.0
    
    def evaluate_coverage(self, original_query: str, multi_queries: List[str]) -> float:
        """
        Evaluate how well the queries cover different aspects of the original query.
        
        Args:
            original_query: The original query
            multi_queries: List of generated queries
            
        Returns:
            Coverage score (0-1)
        """
        queries_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(multi_queries)])
        
        prompt = f"""You are evaluating how well multiple queries cover the aspects of an original query.

Original Query:
{original_query}

Generated Queries:
{queries_text}

Evaluation Criteria:
1. Do the queries collectively address all key aspects of the original query?
2. Are important subtopics or perspectives included?
3. Is the coverage comprehensive without being excessive?

Please evaluate the coverage on a scale from 0 to 1, where:
- 0: Queries miss major aspects of the original query
- 0.5: Queries cover some but not all important aspects
- 1: Queries comprehensively cover all relevant aspects
"""
        
        try:
            result = self.client.chat.completions.create(
                model="gpt-4",
                response_model=EvaluationResult,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for query coverage."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            return result.score
            
        except Exception as e:
            self.console.log(f"[red]Error in coverage evaluation: {str(e)}[/red]")
            return 0.0
    
    def evaluate_relevance(self, original_query: str, multi_queries: List[str]) -> float:
        """
        Evaluate how relevant the generated queries are to the original.
        
        Args:
            original_query: The original query
            multi_queries: List of generated queries
            
        Returns:
            Relevance score (0-1)
        """
        queries_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(multi_queries)])
        
        prompt = f"""You are evaluating the relevance of multiple queries to an original query.

Original Query:
{original_query}

Generated Queries:
{queries_text}

Evaluation Criteria:
1. Are all generated queries directly related to the original query's intent?
2. Do the queries maintain the core topic without drifting?
3. Would answers to these queries help answer the original query?

Please evaluate the relevance on a scale from 0 to 1, where:
- 0: Queries are off-topic or unrelated to the original
- 0.5: Queries are somewhat related but may drift from the original intent
- 1: All queries are highly relevant and aligned with the original intent
"""
        
        try:
            result = self.client.chat.completions.create(
                model="gpt-4",
                response_model=EvaluationResult,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for query relevance."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            return result.score
            
        except Exception as e:
            self.console.log(f"[red]Error in relevance evaluation: {str(e)}[/red]")
            return 0.0
    
    def load_sample_data(self, data_path: str) -> Dict[str, Any]:
        """
        Load multi-query sample data from JSON file.
        
        Args:
            data_path: Path to the sample data file
            
        Returns:
            Dictionary with sample data
        """
        return self.load_json_data(data_path)