"""
Main evaluation module for FunctionChat-Bench integration with BERGEN-UP.

This module provides the entry point for FunctionChat evaluation within
the BERGEN-UP pipeline system.
"""

import os
import json
import logging
from typing import Dict, List, Any
from pathlib import Path

from modules.benchmark.function_chat.evaluation_framework.evaluation_handler import FunctionChatEvaluationHandler


class FunctionChatEvaluation:
    """
    Main evaluation class for FunctionChat-Bench.
    
    This class integrates FunctionChat-Bench evaluation capabilities
    into the BERGEN-UP pipeline system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FunctionChat evaluation.
        
        Args:
            config: Configuration dictionary from BERGEN-UP
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Extract FunctionChat-specific config
        self.function_chat_config = config
        
        # Set default values - path should already be resolved by RAG orchestrator
        self.data_path = self.function_chat_config.get(
            'data_path', 
            'data/benchmark/functionchat_bench'
        )
        
        # Log the resolved path for debugging
        self.logger.info(f"FunctionChat data path: {self.data_path}")
        self.logger.info(f"Data path exists: {os.path.exists(self.data_path)}")
        self.evaluation_types = self.function_chat_config.get(
            'evaluation_types', 
            ['dialog', 'singlecall']
        )
        
    def _load_dataset(self, dataset_type: str) -> List[Dict[str, Any]]:
        """
        Load FunctionChat dataset.
        
        Args:
            dataset_type: Type of dataset ('dialog', 'singlecall', etc.)
            
        Returns:
            List of evaluation examples
        """
        # Check if custom dataset files are specified in config
        dataset_files = self.function_chat_config.get('dataset_files', {})
        if dataset_type in dataset_files:
            dataset_file = dataset_files[dataset_type]
        else:
            # Fallback to default naming
            dataset_file = f"FunctionChat-{dataset_type.capitalize()}.jsonl"
        
        dataset_path = os.path.join(self.data_path, dataset_file)
        
        self.logger.info(f"Loading dataset: {dataset_path}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        dataset = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    example = json.loads(line.strip())
                    # Handle dialog format with turns
                    if 'turns' in example:
                        for turn in example['turns']:
                            turn_example = {
                                'dialog_num': example['dialog_num'],
                                'tools_count': example['tools_count'],
                                'tools': example['tools'],
                                **turn
                            }
                            dataset.append(turn_example)
                    # Handle singlecall format with multiple query items
                    elif 'query' in example and isinstance(example['query'], list):
                        for i, query_item in enumerate(example['query']):
                            # Each query item becomes a separate evaluation example
                            query_example = {
                                'function_num': example.get('function_num'),
                                'function_name': example.get('function_name'),
                                'function_info': example.get('function_info'),
                                'tools': example.get('tools', []),
                                'serial_num': query_item.get('serial_num'),
                                'content': query_item.get('content'),
                                'ground_truth': None,
                                'acceptable_arguments': None
                            }
                            
                            # Find matching ground truth and acceptable arguments
                            if 'ground_truth' in example and isinstance(example['ground_truth'], list):
                                for gt_item in example['ground_truth']:
                                    if gt_item.get('serial_num') == query_item.get('serial_num'):
                                        query_example['ground_truth'] = gt_item.get('content')
                                        break
                            
                            if 'acceptable_arguments' in example and isinstance(example['acceptable_arguments'], list):
                                for acc_item in example['acceptable_arguments']:
                                    if acc_item.get('serial_num') == query_item.get('serial_num'):
                                        query_example['acceptable_arguments'] = acc_item.get('content')
                                        break
                            
                            dataset.append(query_example)
                    else:
                        dataset.append(example)
        
        return dataset
    
    def _prepare_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare evaluation example for processing.
        
        Args:
            example: Raw example from dataset
            
        Returns:
            Processed example
        """
        # Handle different data formats
        if 'content' in example:
            # Singlecall format - single query with content
            ground_truth = example.get('ground_truth', '')
            # Parse ground truth JSON string if it's a string
            if isinstance(ground_truth, str) and ground_truth:
                try:
                    # First parse the outer JSON
                    parsed_gt = json.loads(ground_truth)
                    # The structure should be {"name": "...", "arguments": "..."}
                    # We need to convert this to tool_calls format expected by evaluator
                    if 'name' in parsed_gt and 'arguments' in parsed_gt:
                        # Parse arguments if it's a string
                        args = parsed_gt['arguments']
                        if isinstance(args, str):
                            try:
                                args = json.loads(args) if args.strip() else {}
                            except json.JSONDecodeError:
                                args = {}
                        elif args is None:
                            args = {}
                        
                        # Keep arguments as string for compatibility with evaluation handler
                        ground_truth = {
                            'tool_calls': [{
                                'function': {
                                    'name': parsed_gt['name'],
                                    'arguments': parsed_gt['arguments']  # Keep as string
                                }
                            }]
                        }
                    else:
                        ground_truth = parsed_gt
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to parse ground_truth as JSON: {ground_truth}")
                    ground_truth = {}
            
            # Convert tools from FunctionChat format to OpenAI format
            tools = example.get('tools', [])
            openai_tools = []
            if tools:
                # FunctionChat has tools like [{"type": "exact", "content": [function_list]}]
                # We need to extract and convert to OpenAI format
                for tool_group in tools:
                    if 'content' in tool_group and isinstance(tool_group['content'], list):
                        for tool_item in tool_group['content']:
                            if tool_item.get('type') == 'function' and 'function' in tool_item:
                                openai_tools.append({
                                    'type': 'function',
                                    'function': tool_item['function']
                                })
            
            processed = {
                'serial_num': example.get('serial_num', 0),
                'messages': [{'role': 'user', 'content': example.get('content', '')}],
                'tools': openai_tools,
                'ground_truth': ground_truth,
                'type_of_output': example.get('type_of_output', 'call'),
                'acceptable_arguments': example.get('acceptable_arguments'),
                'temperature': self.function_chat_config.get('temperature', 0.0),
                'tool_choice': self.function_chat_config.get('tool_choice', 'auto')
            }
        else:
            # Dialog format or other formats
            processed = {
                'serial_num': example.get('serial_num', 0),
                'messages': example.get('query', []),
                'tools': example.get('tools', []),
                'ground_truth': example.get('ground_truth', {}),
                'type_of_output': example.get('type_of_output', 'call'),
                'acceptable_arguments': example.get('acceptable_arguments'),
                'temperature': self.function_chat_config.get('temperature', 0.0),
                'tool_choice': self.function_chat_config.get('tool_choice', 'auto')
            }
        
        return processed
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run FunctionChat evaluation.
        
        Returns:
            Evaluation results
        """
        self.logger.info("Starting FunctionChat evaluation")
        
        all_results = {}
        
        for eval_type in self.evaluation_types:
            self.logger.info(f"Running {eval_type} evaluation")
            
            try:
                # Load dataset
                dataset = self._load_dataset(eval_type)
                self.logger.info(f"Loaded {len(dataset)} examples for {eval_type}")
                
                # Prepare examples
                prepared_dataset = [self._prepare_example(ex) for ex in dataset]
                
                # Initialize evaluation handler
                handler = FunctionChatEvaluationHandler(
                    config=self.config,
                    evaluation_type=eval_type
                )
                
                # Run evaluation
                output_path = os.path.join(
                    self.config.get('output_dir', 'outputs'),
                    f'function_chat_{eval_type}_results.jsonl'
                )
                
                results = handler.evaluate_dataset(
                    dataset=prepared_dataset,
                    output_path=output_path,
                    only_exact=self.function_chat_config.get('only_exact', False)
                )
                
                all_results[eval_type] = results
                
                self.logger.info(
                    f"{eval_type} evaluation completed. "
                    f"Accuracy: {results['metrics']['accuracy']:.3f}"
                )
                
            except Exception as e:
                self.logger.error(f"Error in {eval_type} evaluation: {str(e)}")
                all_results[eval_type] = {'error': str(e)}
        
        # Save summary results
        summary_path = os.path.join(
            self.config.get('output_dir', 'outputs'),
            'function_chat_summary.json'
        )
        
        # Convert config to regular dict to avoid serialization issues
        import copy
        
        def convert_to_serializable(obj, seen=None):
            """Recursively convert complex objects to JSON-serializable forms"""
            if seen is None:
                seen = set()
            
            # Prevent infinite recursion
            obj_id = id(obj)
            if obj_id in seen:
                return str(obj)
            seen.add(obj_id)
            
            try:
                # Test if already serializable
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                pass
            
            if obj is None:
                return None
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif hasattr(obj, '_content'):
                return convert_to_serializable(obj._content, seen)
            elif hasattr(obj, 'items'):
                try:
                    return {str(k): convert_to_serializable(v, seen) for k, v in obj.items()}
                except:
                    return str(obj)
            elif isinstance(obj, (list, tuple)):
                try:
                    return [convert_to_serializable(item, seen) for item in obj]
                except:
                    return str(obj)
            else:
                return str(obj)
        
        config_dict = convert_to_serializable(self.function_chat_config)
            
        summary = {
            'config': config_dict,
            'results': {
                eval_type: result.get('metrics', result)
                for eval_type, result in all_results.items()
            }
        }
        
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"FunctionChat evaluation completed. Summary saved to {summary_path}")
        
        return all_results


def run_function_chat_evaluation(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run FunctionChat evaluation with given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Evaluation results
    """
    evaluator = FunctionChatEvaluation(config)
    return evaluator.run_evaluation()