"""
Evaluation handler for FunctionChat-Bench integrated with BERGEN-UP.

This module provides the core evaluation logic for function calling capabilities
using LLM-as-Judge methodology, adapted for BERGEN-UP configuration system.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

from modules.benchmark.function_chat.utils.openai_client import create_openai_client


class FunctionChatEvaluationHandler:
    """
    Main evaluation handler for FunctionChat-Bench evaluation.
    
    This class manages the evaluation process for function calling capabilities
    in conversational contexts using LLM-as-Judge methodology.
    """
    
    def __init__(self, config: Dict[str, Any], evaluation_type: str = "dialog"):
        """
        Initialize the evaluation handler.
        
        Args:
            config: Configuration dictionary from BERGEN-UP
            evaluation_type: Type of evaluation ("dialog", "singlecall", "common")
        """
        self.config = config
        self.evaluation_type = evaluation_type
        self.logger = logging.getLogger(__name__)
        
        # Load rubric prompts
        self.rubric_prompts = self._load_rubric_prompts()
        
        # Initialize LLM client for evaluation
        self.evaluator_client = create_openai_client({
            'llm_model_name': config.get('evaluator_model', 'gpt-4'),
            'llm_api_key': config['llm_api_key'],
            'llm_endpoint': config.get('evaluator_endpoint', 'https://api.openai.com/v1')
        })
        
        # Model under test client
        self.model_client = create_openai_client(config)
        
        self.temperature = config.get('temperature', 0.0)
        
    def _load_rubric_prompts(self) -> Dict[str, str]:
        """Load rubric prompts for different evaluation types."""
        rubric_prompts = {}
        # Use the same data path as configured for datasets
        rubric_path = self.config.get('data_path', 'data/benchmark/functionchat_bench')
        
        for output_type in ['call', 'completion', 'relevance', 'slot']:
            rubric_file_path = os.path.join(rubric_path, f'rubric_{output_type}.txt')
            if os.path.isfile(rubric_file_path):
                with open(rubric_file_path, "r", encoding="utf-8") as f:
                    rubric_prompts[output_type] = f.read().strip()
            else:
                self.logger.warning(f"Rubric file not found: {rubric_file_path}")
                
        return rubric_prompts
    
    def _clean_tool_calls(self, tools: Optional[List[Dict]]) -> Optional[List[Dict]]:
        """Clean tool calls by removing IDs for comparison."""
        if not tools:
            return tools
        
        cleaned_tools = []
        for tool in tools:
            cleaned_tool = tool.copy()
            if 'id' in cleaned_tool:
                del cleaned_tool['id']
            cleaned_tools.append(cleaned_tool)
        return cleaned_tools
    
    def _get_input_prompt(self, inp: Dict[str, Any], out: Dict[str, Any]) -> str:
        """Generate evaluation prompt for LLM-as-Judge."""
        ground_truth = inp['ground_truth'].copy()
        
        # Clean tool calls
        answer_tool_calls = self._clean_tool_calls(ground_truth.get('tool_calls', None))
        if answer_tool_calls:
            ground_truth['tool_calls'] = answer_tool_calls
        out['tool_calls'] = self._clean_tool_calls(out.get('tool_calls', None))
        
        # Get rubric prompt
        output_type = inp['type_of_output']
        rubric_prompt = self.rubric_prompts.get(output_type)
        if not rubric_prompt:
            raise ValueError(f"Unsupported rubric prompt type: {output_type}")
        
        # Format prompt
        tools = json.dumps(inp['tools'], ensure_ascii=False)
        query = json.dumps(inp['messages'], ensure_ascii=False)
        ground_truth_str = json.dumps(ground_truth, ensure_ascii=False)
        response = json.dumps(out, ensure_ascii=False)
        
        if output_type == 'call':
            acceptable_arguments = json.dumps(inp.get('acceptable_arguments', {}), ensure_ascii=False)
            return rubric_prompt.format(
                tools=tools,
                query=query,
                ground_truth=ground_truth_str,
                acceptable_arguments=acceptable_arguments,
                response=response
            )
        elif output_type in ['completion', 'relevance', 'slot']:
            return rubric_prompt.format(
                tools=tools,
                query=query,
                ground_truth=ground_truth_str,
                response=response
            )
        else:
            raise ValueError(f"Unsupported rubric prompt type: {output_type}")
    
    def _get_acceptable_arguments(self, inp: Dict[str, Any]) -> Dict[str, Any]:
        """Parse acceptable arguments from input."""
        acceptable_arguments = inp.get('acceptable_arguments', None)
        if not acceptable_arguments:
            return {}
        
        try:
            if isinstance(acceptable_arguments, str):
                # Handle special cases
                if acceptable_arguments in [
                    "Only ground truth is allowed.",
                    "The date should be expressed as 'tomorrow'. A specific date should not be designated.",
                    "Since the user did not mention a specific year, it will fail if the date was created including the year in the submission."
                ]:
                    return {}
                acceptable_arguments = json.loads(acceptable_arguments)
            return acceptable_arguments
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def _compare_arguments(self, 
                          g_func_args: str, 
                          p_func_args: str, 
                          acceptable_arguments: Dict[str, Any]) -> bool:
        """Compare function arguments with acceptable alternatives."""
        def compare_value(val1, val2):
            if isinstance(val1, str) and isinstance(val2, str):
                val1 = val1.replace(' ', '').lower()
                val2 = val2.replace(' ', '').lower()
            return val1 == val2
        
        if g_func_args == p_func_args:
            return True
        
        try:
            j_g_func_args = json.loads(g_func_args)
            j_p_func_args = json.loads(p_func_args)
        except json.JSONDecodeError:
            return False
        
        # Check for argument hallucination
        for key in j_p_func_args:
            if key not in j_g_func_args:
                return False
        
        # Compare each argument
        pass_arguments = []
        for key, answer in j_g_func_args.items():
            predict = j_p_func_args.get(key, None)
            
            if answer is not None and predict is None:
                return False
            
            if not compare_value(predict, answer):
                if acceptable_arguments and key in acceptable_arguments:
                    acc_values = acceptable_arguments[key]
                    if isinstance(acc_values, list):
                        for acc_answer in acc_values:
                            if compare_value(predict, acc_answer):
                                pass_arguments.append(key)
                                break
                    elif isinstance(acc_values, str):
                        if compare_value(predict, acc_values):
                            pass_arguments.append(key)
            else:
                pass_arguments.append(key)
        
        return len(pass_arguments) == len(j_g_func_args.keys())
    
    def _exact_match(self, inp: Dict[str, Any], out: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Perform exact match evaluation."""
        is_pass = "fail"
        fetch_flag = True
        
        ground_truth = inp.get('ground_truth', {})
        acceptable_arguments = self._get_acceptable_arguments(inp)
        
        if 'tool_calls' in ground_truth:
            ground_truth = ground_truth.get('tool_calls')[0]['function']
        
        g_func_name = ground_truth.get('name')
        g_func_args = ground_truth.get('arguments')
        
        predict_tools = out.get('tool_calls', [])
        diff_case_msg = ''
        
        if predict_tools and len(predict_tools) > 0:
            p_tool = predict_tools[0].get('function', {})
            p_func_name = p_tool.get('name')
            p_func_args = p_tool.get('arguments')
            
            if g_func_name == p_func_name:
                if self._compare_arguments(g_func_args, p_func_args, acceptable_arguments):
                    is_pass = "pass"
                    fetch_flag = False
                else:
                    diff_case_msg += f'g({g_func_args})|p({p_func_args})\\nFunction argument extraction failed.\\n'
            else:
                diff_case_msg += f'g({g_func_name})|p({p_func_name})\\nFunction selection failed.\\n'
        
        msg = f"exact-eval\\n{diff_case_msg}\\n\\n{is_pass}\\n{is_pass}\\n"
        
        evaluate_response = {
            "id": "exact-match",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": msg,
                    "role": "assistant"
                },
                "function_call": None,
                "tool_calls": None,
            }],
            "exact": is_pass
        }
        
        return fetch_flag, evaluate_response
    
    def _llm_evaluate(self, inp: Dict[str, Any], out: Dict[str, Any]) -> Dict[str, Any]:
        """Perform LLM-as-Judge evaluation."""
        input_prompt = self._get_input_prompt(inp, out)
        messages = [{'role': 'user', 'content': input_prompt}]
        
        evaluate_response = self.evaluator_client.predict({
            'temperature': self.temperature,
            'messages': messages
        })
        
        return evaluate_response
    
    def _model_predict(self, inp: Dict[str, Any]) -> Dict[str, Any]:
        """Get model prediction for input."""
        request_data = {
            'messages': inp['messages'],
            'tools': inp.get('tools', []),
            'temperature': inp.get('temperature', self.temperature),
            'tool_choice': inp.get('tool_choice', 'auto')
        }
        
        response = self.model_client.predict(request_data)
        
        # Extract relevant parts
        if response.get('choices'):
            choice = response['choices'][0]
            # Check if we have tool_calls at the choice level (newer format)
            if 'tool_calls' in choice:
                return {
                    'content': choice.get('message', {}).get('content'),
                    'role': choice.get('message', {}).get('role', 'assistant'),
                    'tool_calls': choice.get('tool_calls')
                }
            # Otherwise check in the message (older format)
            elif 'message' in choice:
                message = choice['message']
                return {
                    'content': message.get('content'),
                    'role': message.get('role', 'assistant'),
                    'tool_calls': message.get('tool_calls')
                }
        
        return {'content': None, 'role': 'assistant', 'tool_calls': None}
    
    def evaluate_dataset(self, 
                        dataset: List[Dict[str, Any]], 
                        output_path: str,
                        only_exact: bool = False) -> Dict[str, Any]:
        """
        Evaluate entire dataset.
        
        Args:
            dataset: List of evaluation examples
            output_path: Path to save results
            only_exact: If True, only perform exact match evaluation
            
        Returns:
            Evaluation results summary
        """
        results = []
        
        for example in tqdm(dataset, desc="Evaluating FunctionChat"):
            # Get model prediction
            model_output = self._model_predict(example)
            
            # Determine evaluation type for this example
            example['type_of_output'] = example.get('type_of_output', 'call')
            if self.evaluation_type == 'singlecall':
                example['type_of_output'] = 'call'
            
            # Perform evaluation
            fetch_flag = True
            if example['type_of_output'] == 'call':
                fetch_flag, evaluate_response = self._exact_match(example, model_output)
            
            if not only_exact and fetch_flag:
                evaluate_response = self._llm_evaluate(example, model_output)
            
            # Store result
            result = {
                'serial_num': example.get('serial_num', len(results)),
                'input': example,
                'model_output': model_output,
                'evaluation': evaluate_response,
                'type_of_output': example['type_of_output']
            }
            results.append(result)
        
        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\\n')
        
        # Calculate metrics
        metrics = self._calculate_metrics(results)
        
        return {
            'results': results,
            'metrics': metrics,
            'output_path': output_path
        }
    
    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        total_count = len(results)
        pass_count = 0
        
        for result in results:
            evaluation = result['evaluation']
            if evaluation.get('exact') == 'pass':
                pass_count += 1
            elif 'choices' in evaluation:
                content = evaluation['choices'][0]['message']['content'].lower()
                if 'pass' in content or '패스' in content or '통과' in content:
                    pass_count += 1
        
        accuracy = pass_count / total_count if total_count > 0 else 0.0
        
        return {
            'total_count': total_count,
            'pass_count': pass_count,
            'fail_count': total_count - pass_count,
            'accuracy': accuracy
        }