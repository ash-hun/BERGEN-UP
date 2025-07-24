import uvicorn
import yaml
import tempfile
import os
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from modules.rag import RAG
from config import Evaluation

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

# Pydantic Models for API requests
class PreRetrievalRequest(BaseModel):
    strategies: List[Dict[str, Any]]
    openai_api_key: str
    
class RetrievalRequest(BaseModel):
    strategies: List[Dict[str, Any]] 
    openai_api_key: str
    
class GenerationRequest(BaseModel):
    strategies: List[Dict[str, Any]]
    openai_api_key: str
    
class BenchmarkRequest(BaseModel):
    strategies: List[Dict[str, Any]]
    openai_api_key: str
    benchmark_name: str

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

@app.post("/pre_retrieval")
async def run_pre_retrieval(request: PreRetrievalRequest):
    """Pre-Retrieval 옵션들을 API parameter로 입력받아 결과 도출하는 API"""
    try:
        from modules.pre_retrieval.evaluation import PreRetrievalEvaluation
        
        evaluator = PreRetrievalEvaluation(
            pre_retrieval_strategy=request.strategies,
            openai_api_key=request.openai_api_key
        )
        result = evaluator.run(verbose=False)
        
        return {"status": "success", "result": result, "message": "Pre-Retrieval 평가가 완료되었습니다."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pre-Retrieval 평가 중 오류 발생: {str(e)}")

@app.post("/retrieval")
async def run_retrieval(request: RetrievalRequest):
    """Retrieval 옵션들을 API parameter로 입력받아 결과 도출하는 API"""
    try:
        from modules.retrieval.evaluation import RetrievalEvaluation
        
        evaluator = RetrievalEvaluation(
            retrieval_strategy=request.strategies,
            openai_api_key=request.openai_api_key
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
        
        evaluator = GenerationEvaluation(
            generation_strategy=request.strategies,
            openai_api_key=request.openai_api_key
        )
        result = evaluator.run(verbose=False)
        
        return {"status": "success", "result": result, "message": "Generation 평가가 완료되었습니다."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation 평가 중 오류 발생: {str(e)}")

@app.post("/niah")
async def run_niah_benchmark(request: BenchmarkRequest):
    """NIAH 벤치마크 API"""
    try:
        from modules.benchmark.evaluation import BenchmarkEvaluation
        
        evaluator = BenchmarkEvaluation(
            benchmark_strategy=request.strategies,
            openai_api_key=request.openai_api_key
        )
        result = evaluator.run(verbose=False)
        
        return {"status": "success", "result": result, "message": "NIAH 벤치마크가 완료되었습니다."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NIAH 벤치마크 중 오류 발생: {str(e)}")

@app.post("/function_chat")
async def run_function_chat_benchmark(request: BenchmarkRequest):
    """FunctionChat 벤치마크 API"""
    try:
        from modules.benchmark.function_chat.evaluation import FunctionChatEvaluation
        
        # Convert strategies to expected format
        function_chat_config = {'output_dir': './outputs'}
        for strategy in request.strategies:
            if isinstance(strategy, dict):
                function_chat_config.update(strategy)
        function_chat_config['llm_api_key'] = request.openai_api_key
        
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
