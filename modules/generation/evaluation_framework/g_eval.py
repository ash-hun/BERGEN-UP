from typing import Dict, List, Any
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from rich.console import Console
from tqdm import tqdm

class EvaluationResult(BaseModel):
    """Pydantic model for evaluation results"""
    score: float = Field(ge=0.0, le=1.0, description="Evaluation score between 0 and 1")
    reasoning: str = Field(description="Brief explanation of the evaluation")


class GEval:
    """
    G-Eval implementation for generation evaluation.
    
    Evaluates generation quality using two metrics:
    - Groundedness: How well the answer is grounded in the provided context
    - Answer Relevancy: How relevant the answer is to the question
    """
    
    def __init__(self, openai_api_key: str):
        # Patch OpenAI client with instructor
        self.client = instructor.from_openai(OpenAI(api_key=openai_api_key))
        self.console = Console()
    
    def evaluate_groundedness(self, context: List[str], answer: str) -> float:
        """
        Evaluate how well the answer is grounded in the provided context.
        
        Args:
            context: List of context sentences
            answer: Generated answer
            
        Returns:
            Groundedness score (0-1)
        """
        context_text = "\n".join(context)
        
        prompt = f"""You are evaluating the groundedness of an answer based on the given context.

Context:
{context_text}

Answer:
{answer}

Evaluation Criteria:
1. Is the answer fully supported by the context? (Score 0 if not at all, 1 if completely)
2. Does the answer contain any information not present in the context?
3. Are all claims in the answer verifiable from the context?

Please evaluate the groundedness of the answer on a scale from 0 to 1, where:
- 0: The answer is completely ungrounded (contains information not in context)
- 0.5: The answer is partially grounded (some information from context, some not)
- 1: The answer is fully grounded (all information comes from context)
"""
        
        try:
            result = self.client.chat.completions.create(
                model="gpt-4",
                response_model=EvaluationResult,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for text generation quality."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            return result.score
            
        except Exception as e:
            self.console.log(f"[red]Error in groundedness evaluation: {str(e)}[/red]")
            return 0.0
    
    def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Evaluate how relevant the answer is to the question.
        
        Args:
            question: The question asked
            answer: Generated answer
            
        Returns:
            Relevancy score (0-1)
        """
        prompt = f"""You are evaluating the relevancy of an answer to a given question.

Question:
{question}

Answer:
{answer}

Evaluation Criteria:
1. Does the answer directly address the question asked?
2. Is the answer complete and comprehensive?
3. Does the answer stay focused on the question without unnecessary information?

Please evaluate the relevancy of the answer on a scale from 0 to 1, where:
- 0: The answer is completely irrelevant to the question
- 0.5: The answer partially addresses the question
- 1: The answer perfectly addresses the question
"""
        
        try:
            result = self.client.chat.completions.create(
                model="gpt-4",
                response_model=EvaluationResult,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for text generation quality."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            return result.score
            
        except Exception as e:
            self.console.log(f"[red]Error in answer relevancy evaluation: {str(e)}[/red]")
            return 0.0
    
    def evaluate_batch(self, data: Dict[str, Any], verbose: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a batch of generation results.
        
        Args:
            data: Dictionary containing questions, contexts, and answers
            verbose: Whether to show detailed progress
            
        Returns:
            Dictionary with evaluation scores for each UUID
        """
        results = {}
        uuids = list(data['question'].keys())
        
        # Create progress bar
        with tqdm(total=len(uuids), desc="Evaluating generation", disable=not verbose) as pbar:
            for uuid in uuids:
                question = data['question'][uuid]
                context = data['context'][uuid]
                answer = data['answer'][uuid]
                
                groundedness_score = self.evaluate_groundedness(context, answer)
                relevancy_score = self.evaluate_answer_relevancy(question, answer)
                
                results[uuid] = {
                    'groundedness': groundedness_score,
                    'answer_relevancy': relevancy_score
                }
                
                if verbose:
                    pbar.set_postfix({
                        'UUID': uuid,
                        'G': f"{groundedness_score:.3f}",
                        'R': f"{relevancy_score:.3f}"
                    })
                
                pbar.update(1)
        
        return results