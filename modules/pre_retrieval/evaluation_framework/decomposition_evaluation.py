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


class DecompositionEvaluation(BasePreRetrievalEvaluation):
    """
    Evaluation for Query Decomposition strategy using LLM-as-a-judge.
    
    Evaluates the quality of decomposing complex queries into simpler sub-queries.
    """
    
    def __init__(self, openai_api_key: str):
        super().__init__(openai_api_key)
        self.validate_api_key()
        self.client = instructor.from_openai(OpenAI(api_key=openai_api_key))
    
    def evaluate(self, data: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluate query decomposition quality.
        
        Args:
            data: Dictionary containing complex_query and decomposed_queries
            verbose: Whether to show detailed progress
            
        Returns:
            Dictionary with evaluation results
        """
        results = {}
        uuids = list(data['complex_query'].keys())
        
        with tqdm(total=len(uuids), desc="Evaluating query decomposition", disable=not verbose) as pbar:
            for uuid in uuids:
                complex_query = data['complex_query'][uuid]
                decomposed_queries = data['decomposed_queries'][uuid]
                
                # Evaluate completeness
                completeness_score = self.evaluate_completeness(complex_query, decomposed_queries)
                
                # Evaluate granularity
                granularity_score = self.evaluate_granularity(decomposed_queries)
                
                # Evaluate independence
                independence_score = self.evaluate_independence(decomposed_queries)
                
                # Evaluate answerability
                answerability_score = self.evaluate_answerability(decomposed_queries)
                
                results[uuid] = {
                    'completeness': completeness_score,
                    'granularity': granularity_score,
                    'independence': independence_score,
                    'answerability': answerability_score
                }
                
                if verbose:
                    pbar.set_postfix({
                        'UUID': uuid,
                        'Comp': f"{completeness_score:.3f}",
                        'Gran': f"{granularity_score:.3f}",
                        'Ind': f"{independence_score:.3f}",
                        'Ans': f"{answerability_score:.3f}"
                    })
                
                pbar.update(1)
        
        return results
    
    def evaluate_completeness(self, complex_query: str, decomposed_queries: List[str]) -> float:
        """
        Evaluate whether all aspects of the complex query are covered.
        
        Args:
            complex_query: The original complex query
            decomposed_queries: List of decomposed sub-queries
            
        Returns:
            Completeness score (0-1)
        """
        queries_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(decomposed_queries)])
        
        prompt = f"""You are evaluating the completeness of query decomposition.

Complex Query:
{complex_query}

Decomposed Sub-queries:
{queries_text}

Evaluation Criteria:
1. Are all components and aspects of the complex query addressed?
2. Would answering all sub-queries provide a complete answer to the complex query?
3. Are there any missing elements that should have been decomposed?

Please evaluate the completeness on a scale from 0 to 1, where:
- 0: Major aspects of the complex query are missing
- 0.5: Some important aspects are covered but others are missing
- 1: All aspects are comprehensively covered
"""
        
        try:
            result = self.client.chat.completions.create(
                model="gpt-4",
                response_model=EvaluationResult,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for query decomposition completeness."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            return result.score
            
        except Exception as e:
            self.console.log(f"[red]Error in completeness evaluation: {str(e)}[/red]")
            return 0.0
    
    def evaluate_granularity(self, decomposed_queries: List[str]) -> float:
        """
        Evaluate whether queries are broken down to appropriate level of detail.
        
        Args:
            decomposed_queries: List of decomposed sub-queries
            
        Returns:
            Granularity score (0-1)
        """
        queries_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(decomposed_queries)])
        
        prompt = f"""You are evaluating the granularity of decomposed queries.

Decomposed Sub-queries:
{queries_text}

Evaluation Criteria:
1. Are the sub-queries simple enough to be answered directly?
2. Are they broken down to an appropriate level (not too broad, not too narrow)?
3. Does each sub-query focus on a single, clear aspect?

Please evaluate the granularity on a scale from 0 to 1, where:
- 0: Sub-queries are still too complex or overly fragmented
- 0.5: Some queries have appropriate granularity, others don't
- 1: All queries have optimal granularity for answering
"""
        
        try:
            result = self.client.chat.completions.create(
                model="gpt-4",
                response_model=EvaluationResult,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for query granularity."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            return result.score
            
        except Exception as e:
            self.console.log(f"[red]Error in granularity evaluation: {str(e)}[/red]")
            return 0.0
    
    def evaluate_independence(self, decomposed_queries: List[str]) -> float:
        """
        Evaluate whether sub-queries can be answered independently.
        
        Args:
            decomposed_queries: List of decomposed sub-queries
            
        Returns:
            Independence score (0-1)
        """
        queries_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(decomposed_queries)])
        
        prompt = f"""You are evaluating the independence of decomposed queries.

Decomposed Sub-queries:
{queries_text}

Evaluation Criteria:
1. Can each sub-query be answered without needing answers to other sub-queries?
2. Are the queries self-contained with clear scope?
3. Is there minimal dependency between sub-queries?

Please evaluate the independence on a scale from 0 to 1, where:
- 0: Sub-queries are highly dependent on each other
- 0.5: Some independence but significant dependencies remain
- 1: All sub-queries can be answered independently
"""
        
        try:
            result = self.client.chat.completions.create(
                model="gpt-4",
                response_model=EvaluationResult,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for query independence."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            return result.score
            
        except Exception as e:
            self.console.log(f"[red]Error in independence evaluation: {str(e)}[/red]")
            return 0.0
    
    def evaluate_answerability(self, decomposed_queries: List[str]) -> float:
        """
        Evaluate whether sub-queries are clear and answerable.
        
        Args:
            decomposed_queries: List of decomposed sub-queries
            
        Returns:
            Answerability score (0-1)
        """
        queries_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(decomposed_queries)])
        
        prompt = f"""You are evaluating the answerability of decomposed queries.

Decomposed Sub-queries:
{queries_text}

Evaluation Criteria:
1. Is each sub-query clearly formulated and unambiguous?
2. Can each query be answered with factual information?
3. Are the queries specific enough to have definitive answers?

Please evaluate the answerability on a scale from 0 to 1, where:
- 0: Queries are vague, ambiguous, or unanswerable
- 0.5: Some queries are clear and answerable, others are not
- 1: All queries are clear, specific, and answerable
"""
        
        try:
            result = self.client.chat.completions.create(
                model="gpt-4",
                response_model=EvaluationResult,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for query answerability."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            return result.score
            
        except Exception as e:
            self.console.log(f"[red]Error in answerability evaluation: {str(e)}[/red]")
            return 0.0
    
    def load_sample_data(self, data_path: str) -> Dict[str, Any]:
        """
        Load query decomposition sample data from JSON file.
        
        Args:
            data_path: Path to the sample data file
            
        Returns:
            Dictionary with sample data
        """
        return self.load_json_data(data_path)