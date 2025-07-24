"""
BERGEN-UP API Data Models

This module contains all Pydantic models for API request/response validation.
Organized by module: Pre-Retrieval, Retrieval, Generation, Benchmark.

Each model includes example data based on README.md configurations.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

# =============================================================================
# Pre-Retrieval API Models (Separated by Strategy)
# =============================================================================

# Multi-Query API Models
class MultiQueryRequest(BaseModel):
    path: str = Field(
        example="${hydra:runtime.cwd}/data/pre_retrieval/multi_query/sample_data.json",
        description="Path to multi-query sample data JSON file"
    )
    openai_api_key: str = Field(
        description="OpenAI API key for LLM-as-Judge evaluation"  
    )

    class Config:
        json_schema_extra = {
            "example": {
                "path": "/path/to/data/pre_retrieval/multi_query/sample_data.json",
                "openai_api_key": "sk-your-openai-api-key"
            }
        }

# Query Decomposition API Models  
class QueryDecompositionRequest(BaseModel):
    path: str = Field(
        example="${hydra:runtime.cwd}/data/pre_retrieval/query_decomposition/sample_data.json",
        description="Path to query decomposition sample data JSON file"
    )
    openai_api_key: str = Field(
        description="OpenAI API key for LLM-as-Judge evaluation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "path": "/path/to/data/pre_retrieval/query_decomposition/sample_data.json", 
                "openai_api_key": "sk-your-openai-api-key"
            }
        }

# HyDE API Models
class HyDERequest(BaseModel):
    path: str = Field(
        example="${hydra:runtime.cwd}/data/pre_retrieval/hyde/sample_data.json",
        description="Path to HyDE sample data JSON file"
    )
    openai_api_key: str = Field(
        description="OpenAI API key for LLM-as-Judge evaluation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "path": "/path/to/data/pre_retrieval/hyde/sample_data.json",
                "openai_api_key": "sk-your-openai-api-key"
            }
        }

# =============================================================================
# Retrieval API Models
# =============================================================================

class RetrievalStrategy(BaseModel):
    sample_data_path: str = Field(
        example="${hydra:runtime.cwd}/data/retrieval/sample_data.json",
        description="Path to retrieval sample data JSON file"
    )
    top_k: List[int] = Field(
        default=[1, 3, 5, 10, 20],
        example=[1, 3, 5, 10, 20],
        description="List of k values for which to calculate metrics (e.g., [1, 5, 10])"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "sample_data_path": "/path/to/data/retrieval/sample_data.json",
                "top_k": [1, 3, 5, 10]
            }
        }

class RetrievalRequest(BaseModel):
    strategies: List[RetrievalStrategy] = Field(
        description="List of retrieval strategies to evaluate"
    )
    openai_api_key: str = Field(
        description="OpenAI API key for evaluation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "strategies": [
                    {
                        "sample_data_path": "/path/to/data/retrieval/sample_data.json",
                        "top_k": [1, 5, 10]
                    }
                ],
                "openai_api_key": "sk-your-openai-api-key"
            }
        }

# =============================================================================
# Generation API Models
# =============================================================================

class GEvalConfig(BaseModel):
    mode: str = Field(
        example="standard",
        description="Evaluation mode: 'standard' or 'custom'"
    )
    metric_name: str = Field(
        example="Answer Relevancy",
        description="Name of the metric to evaluate (Answer Relevancy, Consistency, Fluency, Groundness, Relevancy)"
    )
    metric_description: Optional[str] = Field(
        None,
        example="Evaluating how technically accurate and precise the answer is",
        description="Description for custom metrics"
    )
    metric_criterion: Optional[str] = Field(
        None,
        example="- 1: Very Poor. The answer contains significant technical errors.\n- 2: Poor. The answer has some technical accuracy but contains notable errors.\n- 3: Fair. The answer is generally accurate but lacks precision.\n- 4: Good. The answer is technically accurate with minor issues.\n- 5: Excellent. The answer is perfectly accurate and technically precise.",
        description="Evaluation criteria for custom metrics"
    )
    metric_llm: Dict[str, Any] = Field(
        example={
            "model_name": "gpt-4",
            "temperature": 0.0,
            "max_tokens": 1024
        },
        description="LLM configuration for evaluation"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "name": "Standard Metrics Example",
                    "value": {
                        "mode": "standard",
                        "metric_name": "Answer Relevancy",
                        "metric_llm": {
                            "model_name": "gpt-4",
                            "temperature": 0.0,
                            "max_tokens": 1024
                        }
                    }
                },
                {
                    "name": "Custom Metrics Example",
                    "value": {
                        "mode": "custom",
                        "metric_name": "Technical Accuracy",
                        "metric_description": "Evaluating how technically accurate and precise the answer is",
                        "metric_criterion": "- 1: Very Poor. The answer contains significant technical errors.\n- 2: Poor. The answer has some technical accuracy but contains notable errors.\n- 3: Fair. The answer is generally accurate but lacks precision.\n- 4: Good. The answer is technically accurate with minor issues.\n- 5: Excellent. The answer is perfectly accurate and technically precise.",
                        "metric_llm": {
                            "model_name": "gpt-4",
                            "temperature": 0.0,
                            "max_tokens": 1024
                        }
                    }
                }
            ]
        }

class GenerationStrategy(BaseModel):
    sample_data_path: str = Field(
        example="${hydra:runtime.cwd}/data/generation/sample_generation_data.json",
        description="Path to generation sample data JSON file"
    )
    evaluation_metrics: Optional[List[str]] = Field(
        None,
        example=["groundedness", "answer_relevancy"],
        description="List of evaluation metrics to use"
    )
    g_eval_config: Optional[GEvalConfig] = Field(
        None,
        description="G-Eval configuration for generation evaluation"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "name": "Standard Metrics Example",
                    "value": {
                        "sample_data_path": "/path/to/data/generation/sample_generation_data.json",
                        "evaluation_metrics": ["groundedness", "answer_relevancy"],
                        "g_eval_config": {
                            "mode": "standard",
                            "metric_name": "Answer Relevancy",
                            "metric_llm": {
                                "model_name": "gpt-4",
                                "temperature": 0.0,
                                "max_tokens": 1024
                            }
                        }
                    }
                },
                {
                    "name": "Custom Metrics Example",
                    "value": {
                        "sample_data_path": "/path/to/data/generation/sample_generation_data.json",
                        "g_eval_config": {
                            "mode": "custom",
                            "metric_name": "Technical Accuracy",
                            "metric_description": "Evaluating how technically accurate and precise the answer is",
                            "metric_criterion": "- 1: Very Poor. The answer contains significant technical errors.\n- 2: Poor. The answer has some technical accuracy but contains notable errors.\n- 3: Fair. The answer is generally accurate but lacks precision.\n- 4: Good. The answer is technically accurate with minor issues.\n- 5: Excellent. The answer is perfectly accurate and technically precise.",
                            "metric_llm": {
                                "model_name": "gpt-4",
                                "temperature": 0.0,
                                "max_tokens": 1024
                            }
                        }
                    }
                }
            ]
        }

class GenerationRequest(BaseModel):
    strategies: List[GenerationStrategy] = Field(
        description="List of generation strategies to evaluate"
    )
    openai_api_key: str = Field(
        description="OpenAI API key for G-Eval evaluation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "strategies": [
                    {
                        "sample_data_path": "/path/to/data/generation/sample_generation_data.json",
                        "evaluation_metrics": ["groundedness", "answer_relevancy"],
                        "g_eval_config": {
                            "mode": "standard",
                            "metric_name": "Answer Relevancy",
                            "metric_llm": {
                                "model_name": "gpt-4",
                                "temperature": 0.0,
                                "max_tokens": 1024
                            }
                        }
                    }
                ],
                "openai_api_key": "sk-your-openai-api-key"
            }
        }

# =============================================================================
# Benchmark API Models
# =============================================================================

# NIAH Benchmark Models
class NIAHConfig(BaseModel):
    context_lengths: List[int] = Field(
        example=[1000, 2000, 4000],
        description="List of context lengths to test"
    )
    document_depth_percents: List[float] = Field(
        example=[0.1, 0.5, 0.9],
        description="List of document depth percentages to test"
    )
    num_samples_per_test: int = Field(
        default=2,
        example=2,
        description="Number of samples per test case"
    )
    save_results: bool = Field(
        default=True,
        example=True,
        description="Whether to save evaluation results"
    )
    save_contexts: bool = Field(
        default=False,
        example=False,
        description="Whether to save generated contexts"
    )
    test_cases: List[str] = Field(
        example=["single_needle", "multi_needle", "complex_info"],
        description="List of test cases to run"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "name": "Basic NIAH Configuration",
                    "value": {
                        "context_lengths": [1000, 2000, 4000],
                        "document_depth_percents": [0.1, 0.5, 0.9],
                        "num_samples_per_test": 2,
                        "save_results": True,
                        "save_contexts": False,
                        "test_cases": ["single_needle", "multi_needle", "complex_info"]
                    }
                },
                {
                    "name": "Limited NIAH Configuration",
                    "value": {
                        "context_lengths": [1000, 2000],
                        "document_depth_percents": [0.1, 0.5],
                        "num_samples_per_test": 1,
                        "save_results": True,
                        "save_contexts": False,
                        "test_cases": ["single_needle", "multi_needle"]
                    }
                }
            ]
        }

class NIAHStrategy(BaseModel):
    llm_endpoint: str = Field(
        example="openai/gpt-4o",
        description="LLM endpoint to use for evaluation"
    )
    needle_config_path: str = Field(
        example="${hydra:runtime.cwd}/data/benchmark/NIAH/needle_config.json",
        description="Path to needle configuration JSON file"
    )
    NIAH: NIAHConfig = Field(
        description="NIAH evaluation configuration"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "llm_endpoint": "openai/gpt-4o",
                "needle_config_path": "/path/to/data/benchmark/NIAH/needle_config.json",
                "NIAH": {
                    "context_lengths": [1000, 2000, 4000],
                    "document_depth_percents": [0.1, 0.5, 0.9],
                    "num_samples_per_test": 2,
                    "save_results": True,
                    "save_contexts": False,
                    "test_cases": ["single_needle", "multi_needle", "complex_info"]
                }
            }
        }

class NIAHRequest(BaseModel):
    strategies: List[NIAHStrategy] = Field(
        description="List of NIAH benchmark strategies"
    )
    openai_api_key: str = Field(
        description="OpenAI API key for LLM evaluation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "strategies": [
                    {
                        "llm_endpoint": "openai/gpt-4o",
                        "needle_config_path": "/path/to/data/benchmark/NIAH/needle_config.json",
                        "NIAH": {
                            "context_lengths": [1000, 2000],
                            "document_depth_percents": [0.1, 0.5],
                            "num_samples_per_test": 1,
                            "save_results": True,
                            "save_contexts": False,
                            "test_cases": ["single_needle", "multi_needle"]
                        }
                    }
                ],
                "openai_api_key": "sk-your-openai-api-key"
            }
        }

# FunctionChat Benchmark Models
class DatasetFiles(BaseModel):
    dialog: str = Field(
        default="FunctionChat-Dialog-Sample.jsonl",
        example="FunctionChat-Dialog-Sample.jsonl",
        description="Dialog dataset file name"
    )
    singlecall: str = Field(
        default="FunctionChat-Singlecall-Sample.jsonl",
        example="FunctionChat-Singlecall-Sample.jsonl", 
        description="Singlecall dataset file name"
    )

class FunctionChatStrategy(BaseModel):
    llm_model_name: str = Field(
        example="gpt-4o",
        description="Model to evaluate"
    )
    llm_api_key: str = Field(
        example="${common.OPENAI_API_KEY}",
        description="API key for model under test"
    )
    llm_endpoint: str = Field(
        example="https://api.openai.com/v1",
        description="Endpoint for model under test"
    )
    evaluator_model: Optional[str] = Field(
        default="gpt-4",
        example="gpt-4",
        description="Model used as judge"
    )
    evaluator_endpoint: Optional[str] = Field(
        default="https://api.openai.com/v1",
        example="https://api.openai.com/v1",
        description="Endpoint for evaluator"
    )
    evaluation_types: List[str] = Field(
        example=["dialog", "singlecall"],
        description="Types of evaluation to run"
    )
    data_path: str = Field(
        example="${hydra:runtime.cwd}/data/benchmark/functionchat_bench",
        description="Path to FunctionChat data"
    )
    dataset_files: DatasetFiles = Field(
        default_factory=DatasetFiles,
        description="Dataset file names for dialog and singlecall evaluations"
    )
    temperature: Optional[float] = Field(
        default=0.0,
        example=0.0,
        description="Temperature for model predictions"
    )
    tool_choice: Optional[str] = Field(
        default="auto",
        example="auto",
        description="Tool choice strategy"
    )
    only_exact: Optional[bool] = Field(
        default=False,
        example=False,
        description="If true, only run exact match evaluation"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "name": "Basic FunctionChat Configuration",
                    "value": {
                        "llm_model_name": "gpt-4o",
                        "llm_api_key": "sk-your-api-key",
                        "llm_endpoint": "https://api.openai.com/v1",
                        "evaluator_model": "gpt-4",
                        "evaluator_endpoint": "https://api.openai.com/v1",
                        "evaluation_types": ["dialog", "singlecall"],
                        "data_path": "/path/to/data/benchmark/functionchat_bench",
                        "dataset_files": {
                            "dialog": "FunctionChat-Dialog-Sample.jsonl",
                            "singlecall": "FunctionChat-Singlecall-Sample.jsonl"
                        },
                        "temperature": 0.0,
                        "tool_choice": "auto",
                        "only_exact": False
                    }
                },
                {
                    "name": "Local Model Configuration",
                    "value": {
                        "llm_model_name": "llama-3-70b-instruct",
                        "llm_api_key": "dummy-key",
                        "llm_endpoint": "http://localhost:8000/v1",
                        "evaluator_model": "gpt-4",
                        "evaluator_endpoint": "https://api.openai.com/v1",
                        "evaluation_types": ["singlecall"],
                        "data_path": "/path/to/data/benchmark/functionchat_bench",
                        "temperature": 0.0,
                        "tool_choice": "auto",
                        "only_exact": False
                    }
                }
            ]
        }

class FunctionChatRequest(BaseModel):
    strategies: List[FunctionChatStrategy] = Field(
        description="List of FunctionChat benchmark strategies"
    )
    openai_api_key: str = Field(
        description="OpenAI API key for evaluation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "strategies": [
                    {
                        "llm_model_name": "gpt-4o",
                        "llm_api_key": "sk-your-api-key",
                        "llm_endpoint": "https://api.openai.com/v1",
                        "evaluator_model": "gpt-4",
                        "evaluator_endpoint": "https://api.openai.com/v1",
                        "evaluation_types": ["singlecall"],
                        "data_path": "/path/to/data/benchmark/functionchat_bench",
                        "dataset_files": {
                            "dialog": "FunctionChat-Dialog-Sample.jsonl",
                            "singlecall": "FunctionChat-Singlecall-Sample.jsonl"
                        },
                        "temperature": 0.0,
                        "tool_choice": "auto",
                        "only_exact": True
                    }
                ],
                "openai_api_key": "sk-your-openai-api-key"
            }
        }

# =============================================================================
# Common Response Models
# =============================================================================

class APIResponse(BaseModel):
    status: str = Field(
        example="success",
        description="Response status (success/error)"
    )
    message: str = Field(
        example="Evaluation completed successfully",
        description="Human-readable response message"
    )
    result: Optional[Any] = Field(
        None,
        description="Evaluation results data"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Pre-Retrieval evaluation completed successfully",
                "result": {
                    "evaluation_metrics": {
                        "diversity": 0.85,
                        "coverage": 0.92,
                        "relevance": 0.88
                    }
                }
            }
        }

class ErrorResponse(BaseModel):
    detail: str = Field(
        example="Evaluation failed: Invalid API key provided",
        description="Detailed error message"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Pre-Retrieval evaluation failed: Invalid OpenAI API key provided"
            }
        }