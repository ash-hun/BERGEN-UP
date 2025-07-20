"""
NIAH (Needle In A Haystack) evaluation implementation for BERGEN-UP
Based on: https://github.com/Tongji-KGLLM/U-NIAH
"""
import json
import numpy as np
import asyncio
import time
import os
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import requests
from rich.console import Console
from rich.table import Table
from rich.progress import track
from datetime import datetime, timezone
import yaml
import random
from openai import OpenAI
try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None


class NIAHEvaluation:
    """
    Needle In A Haystack benchmark evaluation for long-context comprehension
    Tests the ability of LLMs to find specific information in large contexts
    """
    
    def __init__(self, 
                 llm_endpoint: str,
                 needle_config_path: Optional[str] = None,
                 context_lengths: Optional[List[int]] = None,
                 document_depth_percents: Optional[List[float]] = None,
                 haystack_dir: Optional[str] = None,
                 num_samples_per_test: int = 5,
                 save_results: bool = True,
                 save_contexts: bool = False,
                 test_cases: Optional[List[str]] = None,
                 openai_api_key: Optional[str] = None):
        """
        Initialize NIAH evaluation
        
        Args:
            llm_endpoint: The LLM API endpoint or provider/model format (e.g., "openai/gpt-4o")
            needle_config_path: Path to needle configuration YAML or JSON file
            context_lengths: List of context lengths to test
            document_depth_percents: List of depth percentages (0.0 to 1.0)
            haystack_dir: Directory containing haystack text files
            num_samples_per_test: Number of samples per test configuration
            save_results: Whether to save results to file
            save_contexts: Whether to save generated contexts
            test_cases: List of test case names to run (e.g., ['single_needle', 'multi_needle'])
            openai_api_key: OpenAI API key if using OpenAI models
        """
        self.llm_endpoint = llm_endpoint
        self.openai_api_key = openai_api_key
        self.console = Console()
        
        # Parse LLM endpoint to check if it's OpenAI
        self.is_openai = False
        self.openai_model = None
        if "/" in llm_endpoint:
            provider, model = llm_endpoint.split("/", 1)
            if provider.lower() == "openai":
                self.is_openai = True
                self.openai_model = model
        
        # Default configurations
        # Convert ListConfig to regular Python lists if needed
        def convert_to_list(value):
            """Convert value to list, handling Hydra ListConfig objects"""
            if value is None:
                return None
            try:
                if OmegaConf and hasattr(value, '_metadata'):
                    # Handle Hydra ListConfig objects
                    return OmegaConf.to_container(value)
                elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                    return list(value)
                else:
                    return [value]
            except Exception:
                return [value] if value is not None else None
        
        self.context_lengths = convert_to_list(context_lengths) or [1000, 2000, 4000, 8000, 16000, 32000]
        self.document_depth_percents = convert_to_list(document_depth_percents) or [0.1, 0.3, 0.5, 0.7, 0.9]
            
        self.haystack_dir = haystack_dir or self._get_default_haystack_dir()
        self.num_samples_per_test = num_samples_per_test
        self.save_results = save_results
        self.save_contexts = save_contexts
        self.selected_test_cases = test_cases  # Store selected test cases
        
        # Load needle configurations
        self.needle_cases = self._load_needle_config(needle_config_path)
        
        # Filter test cases if specified
        if self.selected_test_cases:
            filtered_cases = {}
            for case_name in self.selected_test_cases:
                if case_name in self.needle_cases:
                    filtered_cases[case_name] = self.needle_cases[case_name]
                else:
                    self.console.print(f"[yellow]Warning: Test case '{case_name}' not found in config[/yellow]")
            self.needle_cases = filtered_cases
        
        self.results = []
        
    def _get_default_haystack_dir(self) -> str:
        """Get default haystack directory path"""
        # Check if we have sample haystack texts in data directory
        data_dir = Path(__file__).parent.parent.parent.parent / "data" / "benchmark" / "haystack"
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            self._create_sample_haystack(data_dir)
        return str(data_dir)
    
    def _create_sample_haystack(self, data_dir: Path):
        """Create sample haystack texts if they don't exist"""
        sample_texts = [
            "The history of artificial intelligence dates back to ancient myths and stories. "
            "Modern AI research began in the 1950s with pioneers like Alan Turing and John McCarthy. "
            "The field has evolved through various phases including symbolic AI, expert systems, "
            "machine learning, and deep learning. Each phase brought new insights and challenges.",
            
            "Climate change represents one of the most pressing challenges of our time. "
            "Rising global temperatures are causing melting ice caps, rising sea levels, "
            "and extreme weather events. Scientists worldwide are working on solutions "
            "ranging from renewable energy to carbon capture technologies.",
            
            "The human brain contains approximately 86 billion neurons interconnected "
            "through trillions of synapses. This complex network enables consciousness, "
            "memory, emotion, and all cognitive functions. Neuroscientists continue to "
            "uncover the mysteries of how this biological computer operates."
        ]
        
        for i, text in enumerate(sample_texts):
            file_path = data_dir / f"haystack_{i}.txt"
            # Repeat text to make it longer
            full_text = " ".join([text] * 50)
            file_path.write_text(full_text)
    
    def _load_needle_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load needle test cases from configuration file"""
        if config_path and Path(config_path).exists():
            config_file = Path(config_path)
            
            # Support both YAML and JSON files
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    return config
            elif config_file.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config
            else:
                self.console.print(f"[yellow]Warning: Unsupported config file format: {config_file.suffix}[/yellow]")
                return self._get_default_needle_cases()
        else:
            return self._get_default_needle_cases()
    
    def _get_default_needle_cases(self) -> Dict[str, Any]:
        """Get default needle test cases"""
        return {
            'single_needle': {
                'needles': ["The secret code is ALPHA-7234."],
                'question': "What is the secret code?",
                'true_answer': "ALPHA-7234"
            },
            'multi_needle': {
                'needles': [
                    "The meeting will be held in Conference Room B.",
                    "The meeting time is 3:30 PM.",
                    "The meeting date is next Tuesday."
                ],
                'question': "When and where is the meeting?",
                'true_answer': "The meeting will be held in Conference Room B at 3:30 PM next Tuesday."
            }
        }
    
    def _read_haystack_files(self) -> str:
        """Read and combine haystack files"""
        haystack_content = []
        
        if os.path.isdir(self.haystack_dir):
            # Read all text files in directory
            for file_path in sorted(Path(self.haystack_dir).glob("*.txt")):
                with open(file_path, 'r', encoding='utf-8') as f:
                    haystack_content.append(f.read())
        else:
            # Single file
            with open(self.haystack_dir, 'r', encoding='utf-8') as f:
                haystack_content.append(f.read())
        
        return " ".join(haystack_content)
    
    def _encode_and_trim(self, text: str, max_length: int) -> str:
        """Trim text to approximately max_length tokens"""
        # Simple approximation: 1 token ≈ 4 characters
        # In production, use proper tokenizer
        max_chars = max_length * 4
        if len(text) > max_chars:
            return text[:max_chars]
        return text
    
    def _insert_needles(self, context: str, needles: List[str], depth_percent: float) -> str:
        """Insert needles at specified depth in the context"""
        # Split context into sentences
        sentences = context.split('. ')
        if not sentences:
            return context
        
        # Calculate insertion position
        insert_pos = int(len(sentences) * depth_percent)
        insert_pos = max(1, min(insert_pos, len(sentences) - 1))
        
        # Insert needles
        for needle in needles:
            sentences.insert(insert_pos, needle.strip())
            insert_pos += 1  # Insert subsequent needles after the previous one
        
        return '. '.join(sentences)
    
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM endpoint with prompt"""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the question based solely on the provided context. Be concise and direct."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            if self.is_openai:
                # Use OpenAI API directly
                if not self.openai_api_key:
                    raise ValueError("OpenAI API key is required for OpenAI models")
                
                client = OpenAI(api_key=self.openai_api_key)
                
                response = client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    max_tokens=100,
                    temperature=0.1
                )
                
                return response.choices[0].message.content.strip()
            
            else:
                # Use custom endpoint
                request_data = {
                    "messages": messages,
                    "max_tokens": 100,
                    "temperature": 0.1
                }
                
                response = requests.post(
                    self.llm_endpoint,
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                # Handle different response formats
                if isinstance(result, dict):
                    if 'response' in result:
                        return result['response'].strip()
                    elif 'choices' in result and len(result['choices']) > 0:
                        return result['choices'][0].get('message', {}).get('content', '').strip()
                    elif 'content' in result:
                        return result['content'].strip()
                
                return str(result).strip()
            
        except Exception as e:
            self.console.print(f"[red]Error calling LLM: {e}[/red]")
            return ""
    
    def _evaluate_response(self, response: str, true_answer: str, needles: List[str]) -> float:
        """Evaluate if the response contains the expected information"""
        response_lower = response.lower().strip()
        
        # Check for exact answer match
        if true_answer.lower() in response_lower:
            return 1.0
        
        # Check if all needle information is present
        needle_info_found = 0
        for needle in needles:
            # Extract key information from needle
            key_words = [word.lower() for word in needle.split() 
                        if len(word) > 3 and word.isalnum()]
            
            # Check if key words are in response
            if all(word in response_lower for word in key_words[:3]):  # Check first 3 key words
                needle_info_found += 1
        
        # Calculate partial score
        if needle_info_found > 0:
            return needle_info_found / len(needles)
        
        return 0.0
    
    async def run_single_test(self, 
                            test_case: Dict[str, Any],
                            context_length: int, 
                            depth_percent: float) -> Dict[str, Any]:
        """Run a single NIAH test"""
        needles = test_case['needles']
        question = test_case['question']
        true_answer = test_case['true_answer']
        
        # Read haystack
        haystack = self._read_haystack_files()
        
        # Trim to context length
        haystack = self._encode_and_trim(haystack, context_length)
        
        # Insert needles
        context_with_needles = self._insert_needles(haystack, needles, depth_percent)
        
        # Create prompt
        prompt = f"Context:\n{context_with_needles}\n\nQuestion: {question}"
        
        # Call LLM
        start_time = time.time()
        response = await self._call_llm(prompt)
        elapsed_time = time.time() - start_time
        
        # Evaluate response
        score = self._evaluate_response(response, true_answer, needles)
        
        result = {
            'context_length': context_length,
            'depth_percent': depth_percent,
            'needles': needles,
            'question': question,
            'true_answer': true_answer,
            'model_response': response,
            'score': score,
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Save context if requested
        if self.save_contexts:
            result['context'] = context_with_needles
        
        return result
    
    async def run_test_case(self, test_name: str, test_case: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run all tests for a single test case"""
        results = []
        total_tests = len(self.context_lengths) * len(self.document_depth_percents)
        
        self.console.print(f"\n[cyan]Running test case: {test_name}[/cyan]")
        self.console.print(f"Total configurations: {total_tests}")
        
        for context_length in self.context_lengths:
            for depth_percent in self.document_depth_percents:
                # Run multiple samples for each configuration
                sample_scores = []
                
                for sample in range(self.num_samples_per_test):
                    result = await self.run_single_test(test_case, context_length, depth_percent)
                    result['test_name'] = test_name
                    result['sample'] = sample
                    results.append(result)
                    sample_scores.append(result['score'])
                
                # Print average score for this configuration
                avg_score = np.mean(sample_scores)
                self.console.print(
                    f"Context: {context_length}, Depth: {depth_percent:.1%} → "
                    f"Avg Score: {avg_score:.1%}"
                )
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze and aggregate results"""
        # Group results by test case
        test_cases = {}
        for result in results:
            test_name = result['test_name']
            if test_name not in test_cases:
                test_cases[test_name] = []
            test_cases[test_name].append(result)
        
        analysis = {}
        
        for test_name, test_results in test_cases.items():
            # Create results matrix
            results_matrix = {}
            
            for result in test_results:
                context_length = result['context_length']
                depth_percent = result['depth_percent']
                
                if context_length not in results_matrix:
                    results_matrix[context_length] = {}
                
                if depth_percent not in results_matrix[context_length]:
                    results_matrix[context_length][depth_percent] = []
                
                results_matrix[context_length][depth_percent].append(result['score'])
            
            # Calculate average scores
            avg_matrix = {}
            for context_length, depths in results_matrix.items():
                avg_matrix[context_length] = {}
                for depth_percent, scores in depths.items():
                    avg_matrix[context_length][depth_percent] = np.mean(scores)
            
            # Calculate aggregate metrics
            all_scores = [score for depths in results_matrix.values() 
                         for scores in depths.values() for score in scores]
            
            by_context = {}
            for context_length in self.context_lengths:
                context_scores = []
                if context_length in results_matrix:
                    for scores in results_matrix[context_length].values():
                        context_scores.extend(scores)
                by_context[context_length] = np.mean(context_scores) if context_scores else 0.0
            
            by_depth = {}
            for depth_percent in self.document_depth_percents:
                depth_scores = []
                for context_results in results_matrix.values():
                    if depth_percent in context_results:
                        depth_scores.extend(context_results[depth_percent])
                by_depth[depth_percent] = np.mean(depth_scores) if depth_scores else 0.0
            
            analysis[test_name] = {
                'results_matrix': avg_matrix,
                'overall_accuracy': np.mean(all_scores) if all_scores else 0.0,
                'by_context_length': by_context,
                'by_depth_percent': by_depth
            }
        
        return analysis
    
    def display_results(self, analysis: Dict[str, Any]):
        """Display evaluation results in formatted tables"""
        for test_name, test_analysis in analysis.items():
            # Display results matrix
            table = Table(title=f"NIAH Results - {test_name}")
            table.add_column("Context Length", style="cyan")
            
            # Add columns for each depth
            for depth in self.document_depth_percents:
                table.add_column(f"{depth:.0%}", style="green")
            
            # Add rows for each context length
            for context_length in self.context_lengths:
                row = [str(context_length)]
                for depth in self.document_depth_percents:
                    score = test_analysis['results_matrix'].get(context_length, {}).get(depth, 0.0)
                    row.append(f"{score:.1%}")
                table.add_row(*row)
            
            self.console.print(table)
            
            # Display aggregate metrics
            agg_table = Table(title=f"Aggregate Metrics - {test_name}")
            agg_table.add_column("Metric", style="cyan")
            agg_table.add_column("Value", style="green")
            
            agg_table.add_row("Overall Accuracy", f"{test_analysis['overall_accuracy']:.1%}")
            
            self.console.print(agg_table)
    
    async def run(self) -> Dict[str, Any]:
        """Run NIAH evaluation"""
        self.console.print("[bold yellow]Starting NIAH (Needle In A Haystack) Evaluation[/bold yellow]")
        
        all_results = []
        
        # Run each test case
        for test_name, test_case in self.needle_cases.items():
            results = await self.run_test_case(test_name, test_case)
            all_results.extend(results)
        
        # Analyze results
        analysis = self.analyze_results(all_results)
        
        # Display results
        self.display_results(analysis)
        
        # Save results if requested
        if self.save_results:
            output_dir = Path("outputs") / "benchmark" / "niah"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = output_dir / f"niah_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump({
                    'config': {
                        'context_lengths': self.context_lengths,
                        'document_depth_percents': self.document_depth_percents,
                        'num_samples_per_test': self.num_samples_per_test,
                        'llm_endpoint': self.llm_endpoint
                    },
                    'results': all_results,
                    'analysis': analysis
                }, f, indent=2)
            
            self.console.print(f"\n[green]Results saved to: {results_file}[/green]")
        
        self.results = {
            'raw_results': all_results,
            'analysis': analysis
        }
        
        return self.results