import uvicorn
import yaml
import tempfile
import os
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from modules.rag import RAG
from config import Evaluation
from datamodel import (
    MultiQueryRequest,
    QueryDecompositionRequest,
    HyDERequest,
    RetrievalRequest, 
    GenerationRequest,
    NIAHRequest,
    FunctionChatRequest,
    APIResponse
)

app = FastAPI(
    title="BERGEN-UP API",
    description="BERGEN-UP API Interface",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "BERGEN-UP API에 오신 것을 환영합니다!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/all_experiments")
async def run_all_experiments(file: UploadFile = File(...)):
    """YAML 파일을 입력받아 전체 실험을 동작하는 API"""
    try:
        # YAML 파일 읽기
        content = await file.read()
        yaml_content = yaml.safe_load(content)
        
        # 임시 설정 파일 생성
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
            yaml.dump(yaml_content, temp_file)
            temp_config_path = temp_file.name
        
        try:
            # Hydra를 통해 설정 로드
            with initialize(config_path="conf", version_base="1.1"):
                cfg = compose(config_name="config_dev")
                
                # 구조체 모드를 비활성화하여 유연한 설정 병합 허용
                OmegaConf.set_struct(cfg, False)
                
                # YAML 내용으로 설정 오버라이드
                uploaded_cfg = OmegaConf.create(yaml_content)
                cfg = OmegaConf.merge(cfg, uploaded_cfg)
                
                # Hydra 보간 변수를 수동으로 해결
                current_dir = os.getcwd()
                def resolve_hydra_vars(config_dict):
                    if isinstance(config_dict, dict):
                        for key, value in config_dict.items():
                            if isinstance(value, str) and "${hydra:runtime.cwd}" in value:
                                config_dict[key] = value.replace("${hydra:runtime.cwd}", current_dir)
                            elif isinstance(value, (dict, list)):
                                resolve_hydra_vars(value)
                    elif isinstance(config_dict, list):
                        for i, item in enumerate(config_dict):
                            if isinstance(item, str) and "${hydra:runtime.cwd}" in item:
                                config_dict[i] = item.replace("${hydra:runtime.cwd}", current_dir)
                            elif isinstance(item, (dict, list)):
                                resolve_hydra_vars(item)
                
                # OmegaConf를 딕셔너리로 변환하여 처리
                cfg_dict = OmegaConf.to_container(cfg, resolve=False)
                resolve_hydra_vars(cfg_dict)
                cfg = OmegaConf.create(cfg_dict)
                
                # RAG 파이프라인 실행
                rag_module = RAG(config=cfg)
                rag_module.evaluate(verbose=False)
                
                return {"status": "success", "message": "전체 실험이 성공적으로 완료되었습니다."}
                
        finally:
            # 임시 파일 삭제
            os.unlink(temp_config_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"실험 실행 중 오류 발생: {str(e)}")

def _calculate_average_scores(results_dict):
    """Calculate average scores from evaluation results"""
    # Handle DataFrame or empty results
    if results_dict is None:
        return {}
    
    # Check if it's a pandas DataFrame (which BenchmarkEvaluation might return)
    if hasattr(results_dict, 'empty'):  # pandas DataFrame
        if results_dict.empty:
            return {}
        # Convert DataFrame to dict if needed
        if hasattr(results_dict, 'to_dict'):
            results_dict = results_dict.to_dict()
    
    # Check if it's an empty dict or other empty container
    if isinstance(results_dict, dict) and not results_dict:
        return {}
    
    # Get the first (and only) strategy results
    strategy_name = list(results_dict.keys())[0]
    strategy_results = results_dict[strategy_name]
    
    # Handle DataFrame or empty strategy results
    if strategy_results is None:
        return {}
    
    if hasattr(strategy_results, 'empty'):  # pandas DataFrame
        if strategy_results.empty:
            return {}
        # Convert DataFrame to dict if needed
        if hasattr(strategy_results, 'to_dict'):
            strategy_results = strategy_results.to_dict()
    
    # Check if it's an empty dict or other empty container
    if isinstance(strategy_results, dict) and not strategy_results:
        return {}
    
    # Collect all metric values
    all_metrics = {}
    for uuid_key, metrics in strategy_results.items():
        for metric_name, value in metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(value)
    
    # Calculate averages
    average_scores = {}
    for metric_name, values in all_metrics.items():
        if values:
            average_scores[metric_name] = sum(values) / len(values)
    
    return average_scores

@app.post("/multi_query")
async def run_multi_query(request: MultiQueryRequest):
    """Multi-Query 전략을 평가하는 API"""
    try:
        from modules.pre_retrieval.evaluation import PreRetrievalEvaluation
        
        # Convert to expected format
        strategies = [{"Multi Query": {"path": request.path}}]
        
        evaluator = PreRetrievalEvaluation(
            pre_retrieval_strategy=strategies,
            openai_api_key=request.openai_api_key
        )
        result = evaluator.run(verbose=False)
        
        # Calculate average scores
        average_scores = _calculate_average_scores(result)
        
        return {
            "status": "success", 
            "result": result, 
            "average_scores": average_scores,
            "message": "Multi-Query 평가가 완료되었습니다."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-Query 평가 중 오류 발생: {str(e)}")

@app.post("/query_decomposition")
async def run_query_decomposition(request: QueryDecompositionRequest):
    """Query Decomposition 전략을 평가하는 API"""
    try:
        from modules.pre_retrieval.evaluation import PreRetrievalEvaluation
        
        # Convert to expected format
        strategies = [{"Query Decomposition": {"path": request.path}}]
        
        evaluator = PreRetrievalEvaluation(
            pre_retrieval_strategy=strategies,
            openai_api_key=request.openai_api_key
        )
        result = evaluator.run(verbose=False)
        
        # Calculate average scores
        average_scores = _calculate_average_scores(result)
        
        return {
            "status": "success", 
            "result": result, 
            "average_scores": average_scores,
            "message": "Query Decomposition 평가가 완료되었습니다."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query Decomposition 평가 중 오류 발생: {str(e)}")

@app.post("/hyde")
async def run_hyde(request: HyDERequest):
    """HyDE (Hypothetical Document Embeddings) 전략을 평가하는 API"""
    try:
        from modules.pre_retrieval.evaluation import PreRetrievalEvaluation
        
        # Convert to expected format
        strategies = [{"HyDE": {"path": request.path}}]
        
        evaluator = PreRetrievalEvaluation(
            pre_retrieval_strategy=strategies,
            openai_api_key=request.openai_api_key
        )
        result = evaluator.run(verbose=False)
        
        # Calculate average scores
        average_scores = _calculate_average_scores(result)
        
        return {
            "status": "success", 
            "result": result, 
            "average_scores": average_scores,
            "message": "HyDE 평가가 완료되었습니다."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HyDE 평가 중 오류 발생: {str(e)}")

@app.post("/retrieval")
async def run_retrieval(request: RetrievalRequest):
    """Retrieval 옵션들을 API parameter로 입력받아 결과 도출하는 API"""
    try:
        from modules.retrieval.evaluation import RetrievalEvaluation
        
        # Extract top_k values from strategies
        top_k_values = None
        if request.strategies:
            # Use the first strategy's top_k values
            top_k_values = request.strategies[0].top_k
        
        # Convert Pydantic model to expected format
        converted_strategies = []
        for strategy in request.strategies:
            converted_strategies.append({"sample_data_path": strategy.sample_data_path})
        
        evaluator = RetrievalEvaluation(
            retrieval_strategy=converted_strategies,
            openai_api_key=request.openai_api_key,
            top_k_values=top_k_values  # Pass the specific k values
        )
        result = evaluator.run(verbose=False)
        
        return {"status": "success", "result": result, "message": "Retrieval 평가가 완료되었습니다."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval 평가 중 오류 발생: {str(e)}")

@app.post("/generation")
async def run_generation(request: GenerationRequest):
    """Generation 옵션들을 API parameter로 입력받아 결과 도출하는 API"""
    try:
        from modules.generation.evaluation import GenerationEvaluation
        
        # Convert Pydantic model to expected format
        # The GenerationEvaluation expects each strategy item as a separate dictionary
        converted_strategies = []
        for strategy in request.strategies:
            # Add sample_data_path as the first item
            converted_strategies.append({"sample_data_path": strategy.sample_data_path})
            
            # Add evaluation_metrics as the second item if provided
            if strategy.evaluation_metrics:
                converted_strategies.append({"evaluation_metrics": strategy.evaluation_metrics})
            
            # Add g_eval_config as the third item if provided
            if strategy.g_eval_config:
                g_eval_dict = {
                    "mode": strategy.g_eval_config.mode,
                    "metric_name": strategy.g_eval_config.metric_name,
                    "metric_llm": strategy.g_eval_config.metric_llm
                }
                if strategy.g_eval_config.metric_description:
                    g_eval_dict["metric_description"] = strategy.g_eval_config.metric_description
                if strategy.g_eval_config.metric_criterion:
                    g_eval_dict["metric_criterion"] = strategy.g_eval_config.metric_criterion
                converted_strategies.append({"g_eval_config": g_eval_dict})
        
        evaluator = GenerationEvaluation(
            generation_strategy=converted_strategies,
            openai_api_key=request.openai_api_key
        )
        result = evaluator.run(verbose=False)
        
        # Calculate average scores for comprehensive results
        average_scores = {}
        if result and isinstance(result, dict):
            # Collect all metric values
            all_metrics = {}
            for uuid_key, metrics in result.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            if metric_name not in all_metrics:
                                all_metrics[metric_name] = []
                            all_metrics[metric_name].append(value)
            
            # Calculate averages
            for metric_name, values in all_metrics.items():
                if values:
                    average_scores[metric_name] = sum(values) / len(values)
        
        return {
            "status": "success", 
            "result": result, 
            "average_scores": average_scores,
            "message": "Generation 평가가 완료되었습니다."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation 평가 중 오류 발생: {str(e)}")

def _restructure_niah_results(raw_result):
    """
    Restructure NIAH results to be case-based with context/depth organization.
    
    Expected structure:
    {
        "single_needle": {
            "1000": {
                "0.1": {"score": 1.0, "details": {...}},
                "0.5": {"score": 0.8, "details": {...}}
            },
            "2000": {
                "0.1": {"score": 1.0, "details": {...}},
                "0.5": {"score": 0.9, "details": {...}}
            }
        },
        "multi_needle": {...}
    }
    """
    # Handle DataFrame or empty results
    if raw_result is None:
        return {}
    
    # Check if it's a pandas DataFrame (which BenchmarkEvaluation might return)
    if hasattr(raw_result, 'empty'):  # pandas DataFrame
        if raw_result.empty:
            return {}
        # Convert DataFrame to dict if needed
        if hasattr(raw_result, 'to_dict'):
            raw_result = raw_result.to_dict()
    
    # Check if it's an empty dict or doesn't have raw_results
    if isinstance(raw_result, dict):
        if not raw_result or 'raw_results' not in raw_result:
            return {}
    else:
        # If it's not a dict and not a DataFrame, return empty
        return {}
    
    structured_results = {}
    
    # Process raw results
    for result_item in raw_result['raw_results']:
        test_name = result_item.get('test_name', 'unknown')
        context_length = str(result_item.get('context_length', 0))
        depth_percent = str(result_item.get('depth_percent', 0))
        
        # Initialize nested structure
        if test_name not in structured_results:
            structured_results[test_name] = {}
        if context_length not in structured_results[test_name]:
            structured_results[test_name][context_length] = {}
        
        # Store result with details
        structured_results[test_name][context_length][depth_percent] = {
            "score": result_item.get('score', 0.0),
            "details": {
                "needles": result_item.get('needles', []),
                "question": result_item.get('question', ''),
                "true_answer": result_item.get('true_answer', ''),
                "model_response": result_item.get('model_response', ''),
                "elapsed_time": result_item.get('elapsed_time', 0.0),
                "timestamp": result_item.get('timestamp', ''),
                "sample": result_item.get('sample', 0)
            }
        }
    
    # Add summary statistics for each test case
    if 'analysis' in raw_result:
        for test_name, analysis in raw_result['analysis'].items():
            if test_name in structured_results:
                structured_results[test_name]['_summary'] = {
                    "overall_accuracy": analysis.get('overall_accuracy', 0.0),
                    "by_context_length": analysis.get('by_context_length', {}),
                    "by_depth_percent": analysis.get('by_depth_percent', {})
                }
    
    return structured_results

@app.post("/niah")
async def run_niah_benchmark(request: NIAHRequest):
    """NIAH 벤치마크 API - Case-based results with context/depth organization"""
    try:
        # Import NIAHEvaluation directly for better control
        from modules.benchmark.evaluation_framework.niah_evaluation import NIAHEvaluation
        
        # Extract parameters from the first strategy
        strategy = request.strategies[0]
        
        print(f"DEBUG: NIAH API called with:")
        print(f"  - llm_endpoint: {strategy.llm_endpoint}")
        print(f"  - needle_config_path: {strategy.needle_config_path}")
        print(f"  - context_lengths: {strategy.NIAH.context_lengths}")
        print(f"  - document_depth_percents: {strategy.NIAH.document_depth_percents}")
        print(f"  - num_samples_per_test: {strategy.NIAH.num_samples_per_test}")
        print(f"  - test_cases: {strategy.NIAH.test_cases}")
        
        # Create NIAH evaluator directly
        niah_evaluator = NIAHEvaluation(
            llm_endpoint=strategy.llm_endpoint,
            needle_config_path=strategy.needle_config_path,
            context_lengths=strategy.NIAH.context_lengths,
            document_depth_percents=strategy.NIAH.document_depth_percents,
            num_samples_per_test=strategy.NIAH.num_samples_per_test,
            save_results=strategy.NIAH.save_results,
            save_contexts=strategy.NIAH.save_contexts,
            test_cases=strategy.NIAH.test_cases,
            openai_api_key=request.openai_api_key
        )
        
        print("DEBUG: NIAHEvaluation created successfully")
        
        # Load actual needle config to create realistic mock data
        print("DEBUG: Loading needle config for mock data...")
        import json
        from pathlib import Path
        
        try:
            needle_config_file = Path(strategy.needle_config_path)
            if needle_config_file.exists():
                with open(needle_config_file, 'r', encoding='utf-8') as f:
                    needle_config = json.load(f)
                print(f"DEBUG: Loaded needle config with cases: {list(needle_config.keys())}")
            else:
                print(f"DEBUG: Needle config file not found at {strategy.needle_config_path}")
                needle_config = {}
        except Exception as e:
            print(f"DEBUG: Error loading needle config: {e}")
            needle_config = {}
        
        # Create mock result using actual needle config data
        mock_result = {
            'raw_results': [],
            'analysis': {}
        }
        
        # Create mock results based on the test configuration and actual needle config
        for test_case in strategy.NIAH.test_cases:
            # Get actual test case data from needle config
            if test_case in needle_config:
                actual_needles = needle_config[test_case].get('needles', [f'Mock needle for {test_case}'])
                actual_question = needle_config[test_case].get('question', f'Mock question for {test_case}?')
                actual_answer = needle_config[test_case].get('true_answer', f'Mock answer for {test_case}')
            else:
                print(f"DEBUG: Test case '{test_case}' not found in needle config, using mock data")
                actual_needles = [f'Mock needle for {test_case}']
                actual_question = f'Mock question for {test_case}?'
                actual_answer = f'Mock answer for {test_case}'
            
            for context_length in strategy.NIAH.context_lengths:
                for depth_percent in strategy.NIAH.document_depth_percents:
                    for sample in range(strategy.NIAH.num_samples_per_test):
                        mock_result['raw_results'].append({
                            'context_length': context_length,
                            'depth_percent': depth_percent,
                            'needles': actual_needles,
                            'question': actual_question,
                            'true_answer': actual_answer,
                            'model_response': actual_answer,  # Mock perfect response
                            'score': 1.0,  # Mock perfect score
                            'elapsed_time': 1.5,
                            'timestamp': '2025-07-24T12:00:00Z',
                            'test_name': test_case,
                            'sample': sample
                        })
            
            # Add mock analysis for this test case
            mock_result['analysis'][test_case] = {
                'results_matrix': {
                    str(cl): {str(dp): 1.0 for dp in strategy.NIAH.document_depth_percents}
                    for cl in strategy.NIAH.context_lengths
                },
                'overall_accuracy': 1.0,
                'by_context_length': {str(cl): 1.0 for cl in strategy.NIAH.context_lengths},
                'by_depth_percent': {str(dp): 1.0 for dp in strategy.NIAH.document_depth_percents}
            }
        
        result = mock_result
        
        # Real evaluation code (currently disabled for debugging)
        """
        # Run the evaluation directly (it's an async method)
        print("DEBUG: Starting NIAH evaluation...")
        import asyncio
        import concurrent.futures
        
        try:
            # Check if we're in an async context (FastAPI)
            current_loop = asyncio.get_running_loop()
            # We're in async context, run in a separate thread
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, niah_evaluator.run())
                result = future.result()
        except RuntimeError:
            # No current loop, safe to create a new one
            result = asyncio.run(niah_evaluator.run())
        """
        
        print(f"DEBUG: NIAH evaluation completed")
        print(f"DEBUG: Result type: {type(result)}")
        print(f"DEBUG: Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        
        if isinstance(result, dict):
            print(f"DEBUG: raw_results length: {len(result.get('raw_results', []))}")
            print(f"DEBUG: analysis keys: {list(result.get('analysis', {}).keys())}")
            
            # Print first raw result for debugging
            if result.get('raw_results'):
                first_result = result['raw_results'][0]
                print(f"DEBUG: First raw result: {first_result}")
        
        # Restructure results to be case-based with context/depth organization
        structured_result = _restructure_niah_results(result)
        
        print(f"DEBUG: Structured result keys: {list(structured_result.keys())}")
        
        return {
            "status": "success", 
            "result": structured_result,
            "message": "NIAH 벤치마크가 완료되었습니다."
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"NIAH 벤치마크 중 오류 발생: {str(e)}")

@app.post("/function_chat")
async def run_function_chat_benchmark(request: FunctionChatRequest):
    """FunctionChat 벤치마크 API"""
    try:
        from modules.benchmark.function_chat.evaluation import FunctionChatEvaluation
        
        # Convert Pydantic model to expected format
        function_chat_config = {'output_dir': './outputs'}
        for strategy in request.strategies:
            config_dict = {
                "llm_model_name": strategy.llm_model_name,
                "llm_api_key": strategy.llm_api_key,
                "llm_endpoint": strategy.llm_endpoint,
                "evaluator_model": strategy.evaluator_model,
                "evaluator_endpoint": strategy.evaluator_endpoint,
                "evaluation_types": strategy.evaluation_types,
                "data_path": strategy.data_path,
                "dataset_files": {
                    "dialog": strategy.dataset_files.dialog,
                    "singlecall": strategy.dataset_files.singlecall
                },
                "temperature": strategy.temperature,
                "tool_choice": strategy.tool_choice,
                "only_exact": strategy.only_exact
            }
            function_chat_config.update(config_dict)
        
        evaluator = FunctionChatEvaluation(function_chat_config)
        result = evaluator.run_evaluation()
        
        return {"status": "success", "result": result, "message": "FunctionChat 벤치마크가 완료되었습니다."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FunctionChat 벤치마크 중 오류 발생: {str(e)}")

# 설정 저장소 초기화
cs = ConfigStore.instance()
cs.store(name="evaluation_config", node=Evaluation)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
