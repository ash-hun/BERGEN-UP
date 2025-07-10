from typing import Dict, Any
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm
from .base_evaluation import BasePreRetrievalEvaluation


class EvaluationResult(BaseModel):
    """Pydantic model for evaluation results"""
    score: float = Field(ge=0.0, le=1.0, description="Evaluation score between 0 and 1")
    reasoning: str = Field(description="Brief explanation of the evaluation")


class HyDEEvaluation(BasePreRetrievalEvaluation):
    """
    Evaluation for HyDE (Hypothetical Document Embeddings) strategy using LLM-as-a-judge.
    
    Evaluates the quality of generated hypothetical documents.
    """
    
    def __init__(self, openai_api_key: str):
        super().__init__(openai_api_key)
        self.validate_api_key()
        self.client = instructor.from_openai(OpenAI(api_key=openai_api_key))
    
    def evaluate(self, data: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluate HyDE generation quality.
        
        Args:
            data: Dictionary containing query and hyde_documents
            verbose: Whether to show detailed progress
            
        Returns:
            Dictionary with evaluation results
        """
        results = {}
        uuids = list(data['query'].keys())
        
        with tqdm(total=len(uuids), desc="Evaluating HyDE generation", disable=not verbose) as pbar:
            for uuid in uuids:
                query = data['query'][uuid]
                hyde_document = data['hyde_documents'][uuid]
                
                # Evaluate relevance
                relevance_score = self.evaluate_relevance(query, hyde_document)
                
                # Evaluate specificity
                specificity_score = self.evaluate_specificity(query, hyde_document)
                
                # Evaluate factuality
                factuality_score = self.evaluate_factuality(hyde_document)
                
                # Evaluate coherence
                coherence_score = self.evaluate_coherence(hyde_document)
                
                results[uuid] = {
                    'relevance': relevance_score,
                    'specificity': specificity_score,
                    'factuality': factuality_score,
                    'coherence': coherence_score
                }
                
                if verbose:
                    pbar.set_postfix({
                        'UUID': uuid,
                        'R': f"{relevance_score:.3f}",
                        'S': f"{specificity_score:.3f}",
                        'F': f"{factuality_score:.3f}",
                        'C': f"{coherence_score:.3f}"
                    })
                
                pbar.update(1)
        
        return results
    
    def evaluate_relevance(self, query: str, hyde_document: str) -> float:
        """
        Evaluate how relevant the hypothetical document is to the query.
        
        Args:
            query: The original query
            hyde_document: Generated hypothetical document
            
        Returns:
            Relevance score (0-1)
        """
        prompt = f"""You are evaluating the relevance of a hypothetical document to a query.

Query:
{query}

Hypothetical Document:
{hyde_document}

Evaluation Criteria:
1. Does the document directly address the query?
2. Does it contain information that would help answer the query?
3. Is the content aligned with what the query is asking for?

Please evaluate the relevance on a scale from 0 to 1, where:
- 0: The document is completely irrelevant to the query
- 0.5: The document is partially relevant but misses key aspects
- 1: The document is highly relevant and directly addresses the query
"""
        
        try:
            result = self.client.chat.completions.create(
                model="gpt-4",
                response_model=EvaluationResult,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for document relevance."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            return result.score
            
        except Exception as e:
            self.console.log(f"[red]Error in relevance evaluation: {str(e)}[/red]")
            return 0.0
    
    def evaluate_specificity(self, query: str, hyde_document: str) -> float:
        """
        Evaluate how specific and detailed the hypothetical document is.
        
        Args:
            query: The original query
            hyde_document: Generated hypothetical document
            
        Returns:
            Specificity score (0-1)
        """
        prompt = f"""You are evaluating the specificity of a hypothetical document.

Query:
{query}

Hypothetical Document:
{hyde_document}

Evaluation Criteria:
1. Does the document provide specific details rather than vague statements?
2. Are there concrete examples, data, or technical details?
3. Is the level of detail appropriate for answering the query?

Please evaluate the specificity on a scale from 0 to 1, where:
- 0: The document is vague and lacks specific information
- 0.5: The document has some specifics but could be more detailed
- 1: The document is rich in specific, relevant details
"""
        
        try:
            result = self.client.chat.completions.create(
                model="gpt-4",
                response_model=EvaluationResult,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for document specificity."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            return result.score
            
        except Exception as e:
            self.console.log(f"[red]Error in specificity evaluation: {str(e)}[/red]")
            return 0.0
    
    def evaluate_factuality(self, hyde_document: str) -> float:
        """
        Evaluate the factual accuracy of the hypothetical document.
        
        Args:
            hyde_document: Generated hypothetical document
            
        Returns:
            Factuality score (0-1)
        """
        prompt = f"""You are evaluating the factual accuracy of a hypothetical document.

Hypothetical Document:
{hyde_document}

Evaluation Criteria:
1. Are the statements in the document factually plausible?
2. Does it avoid obvious factual errors or contradictions?
3. Is the information consistent with general knowledge?

Note: Since this is a hypothetical document, evaluate based on plausibility and consistency rather than verified facts.

Please evaluate the factuality on a scale from 0 to 1, where:
- 0: The document contains obvious factual errors or implausible claims
- 0.5: The document is mostly plausible with some questionable elements
- 1: The document is highly plausible and factually consistent
"""
        
        try:
            result = self.client.chat.completions.create(
                model="gpt-4",
                response_model=EvaluationResult,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for document factuality."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            return result.score
            
        except Exception as e:
            self.console.log(f"[red]Error in factuality evaluation: {str(e)}[/red]")
            return 0.0
    
    def evaluate_coherence(self, hyde_document: str) -> float:
        """
        Evaluate how coherent and well-structured the document is.
        
        Args:
            hyde_document: Generated hypothetical document
            
        Returns:
            Coherence score (0-1)
        """
        prompt = f"""You are evaluating the coherence of a hypothetical document.

Hypothetical Document:
{hyde_document}

Evaluation Criteria:
1. Is the document well-structured with logical flow?
2. Are ideas connected coherently throughout?
3. Is the writing clear and easy to understand?

Please evaluate the coherence on a scale from 0 to 1, where:
- 0: The document is disjointed and difficult to follow
- 0.5: The document has some structure but lacks flow in places
- 1: The document is highly coherent with excellent structure and flow
"""
        
        try:
            result = self.client.chat.completions.create(
                model="gpt-4",
                response_model=EvaluationResult,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for document coherence."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            return result.score
            
        except Exception as e:
            self.console.log(f"[red]Error in coherence evaluation: {str(e)}[/red]")
            return 0.0
    
    def load_sample_data(self, data_path: str) -> Dict[str, Any]:
        """
        Load HyDE sample data from JSON file.
        
        Args:
            data_path: Path to the sample data file
            
        Returns:
            Dictionary with sample data
        """
        return self.load_json_data(data_path)