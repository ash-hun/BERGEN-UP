# BERGEN-UP

>   All mighty tool for RAG like ✨BERGEN UP✨

[**BERGEN**](https://github.com/naver/bergen?tab=readme-ov-file) (*BEnchmarking Retrieval-augmented GENeration*) is a library designed to benchmark RAG systems with a focus on question-answering (QA) by **NAVER Labs**. It addresses the challenge of inconsistent benchmarking in comparing approaches and understanding the impact of each component in a RAG pipeline. Unlike BERGEN, BERGEN-UP is an end-to-end evaluation pipeline that enhanced focuses on the diversity of RAG pipelines and the functionality of each modules.

## 🥑 What is support to `BERGEN-UP`?
- RAG pipeline experiment management using YAML configuration files
- Support to evaluate APIs for each sub-module


## 🍒 Key Feature
- **BERGEN-UP Pipeline** 
    - *Chunking*
        - token level
            - recall
            - precision
            - iou
    - *Pre-Retrieval*
        - multi-query
        - decomposition
        - hyde
    - *Retrieval*
        - evaluation level
            - precision@k
            - recall@k  
            - f1@k
            - ndcg@k
            - hit_rate@k
            - mrr
    - *Generation*
        - standard metrics (G-Eval)
            - groundedness
            - answer_relevancy
            - consistency
            - fluency
            - relevancy
        - custom metrics
- **BENCHMARK Pipeline**
    - *Bench-Test*
        - U-NIAH (Needle in the haystack)
            - selective sub-modules
            - JSON config support
        - FunctionChat Bench
            - Korean function calling evaluation
            - LLM-as-Judge methodology
            - Local model support
        <!-- - BEIR
        - ASQA
        - TriviaQA
        - HotpotQA
        - WikiQA
        - NQ -->
<!-- - **Extra Module** for RAG
    - Generate Synthetic Dataset
        - QA (= Question Answering) -->


## 🍑 How to run pipeline?

##### 1. Write your evaluation in `conf/config.yaml`

##### 2. Run only below script
```bash
$ uv run pipeline.py label='__experiments_name__'
```

## 🍊 Core points Each Module

<details>
<summary>Chunking Module</summary>

- 핵심 기능
    - Token Level 평가
        - Metric : (https://research.trychroma.com/evaluating-chunking)
            - iou
            - precision
            - recall

- 사용법
    - `conf/config.yaml`의 `chunking` 섹션에 아래 내용을 참고하여 작성한다.
    ```yaml
    chunking:
        strategies: 
            - question_set_path: "${hydra:runtime.cwd}/data/chunking/question_set/questions_df_chatlogs.csv"
            - corpora_id_paths:
                chatlogs: "${hydra:runtime.cwd}/data/chunking/corpora/chatlogs.md"
            - Semantic Chunking:
                mode: openai
                embedding_model: "text-embedding-3-large"
                custom_url: "custom_embedding_function_api_address"
            - Recursive Token Chunking:
                chunk_size: 800
                chunk_overlap: 400
            - Fixed Token Chunking:
                chunk_size: 800
                chunk_overlap: 400
    ```

</details>

<details>
<summary>Pre-Retrieval Module</summary>

- 핵심 기능
    - LLM-as-a-Judge 기반 품질 평가
        - Multi-Query 평가 지표:
            - diversity : 생성된 다중 쿼리들 간의 다양성 평가 (0-1)
            - coverage : 원본 쿼리의 다양한 측면을 얼마나 포괄하는지 평가 (0-1)
            - relevance : 생성된 쿼리들이 원본 쿼리와 얼마나 관련성이 있는지 평가 (0-1)
        - Query Decomposition 평가 지표:
            - completeness : 복잡한 쿼리를 얼마나 완전하게 분해했는지 평가 (0-1)
            - granularity : 분해된 쿼리들의 적절한 세분화 정도 평가 (0-1)
            - independence : 각 분해된 쿼리가 독립적으로 답변 가능한지 평가 (0-1)
            - answerability : 분해된 쿼리들이 실제로 답변 가능한지 평가 (0-1)
        - HyDE (Hypothetical Document Embeddings) 평가 지표:
            - relevance : 생성된 가상 문서가 쿼리와 얼마나 관련성이 있는지 평가 (0-1)
            - specificity : 생성된 문서가 얼마나 구체적이고 상세한지 평가 (0-1)
            - factuality : 생성된 문서의 사실적 정확성 평가 (0-1)
            - coherence : 생성된 문서의 일관성과 논리적 흐름 평가 (0-1)

- 사용법
    - `conf/config.yaml`의 `pre_retrieval` 섹션에 아래 내용을 참고하여 작성한다.
    ```yaml
    pre_retrieval:
        strategies: 
            - Multi Query:
                path: "${hydra:runtime.cwd}/data/pre_retrieval/multi_query/sample_data.json"
            - Query Decomposition:
                path: "${hydra:runtime.cwd}/data/pre_retrieval/query_decomposition/sample_data.json"
            - HyDE:
                path: "${hydra:runtime.cwd}/data/pre_retrieval/hyde/sample_data.json"
    ```

</details>

<details>
<summary>Retrieval Module</summary>

- 핵심 기능
    - Evaluation Level 평가
        - Metric : 
            - precision@k : 검색된 상위 k개 결과 중 관련 문서의 비율
            - recall@k : 전체 관련 문서 중 상위 k개 결과에서 검색된 비율
            - f1@k : precision@k와 recall@k의 조화평균
            - ndcg@k : 순위를 고려한 누적 할인 게인
            - hit_rate@k : 상위 k개 결과에 관련 문서가 하나라도 있는 비율
            - mrr : 첫 번째 관련 문서의 순위 역수 평균

- 사용법
    - `conf/config.yaml`의 `retrieval` 섹션에 아래 내용을 참고하여 작성한다.
    ```yaml
    retrieval:
        strategies: 
            - sample_data_path: "${hydra:runtime.cwd}/data/retrieval/sample_data.json"
            - top_k: 10
    ```

</details>

<details>
<summary>Generation Module</summary>

- 핵심 기능
    - G-Eval 기반 생성 품질 평가
        - Standard Metrics (표준 평가 지표):
            - groundedness : 생성된 답변이 제공된 컨텍스트에 얼마나 근거하는지 평가 (0-1)
            - answer_relevancy : 생성된 답변이 질문에 얼마나 관련성이 있는지 평가 (0-1)
            - consistency : 생성된 답변의 내부 일관성 평가 (0-1)
            - fluency : 생성된 답변의 유창성 및 가독성 평가 (0-1)
            - relevancy : 검색된 컨텍스트가 질문에 얼마나 관련성이 있는지 평가 (0-1)
        - Custom Metrics (사용자 정의 평가 지표):
            - 사용자가 정의한 평가 기준에 따른 맞춤형 평가 가능
            - 1-5 점 척도로 세밀한 평가 지원

- 사용법
    - `conf/config.yaml`의 `generation` 섹션에 아래 내용을 참고하여 작성한다.
    
    **기본 사용법 (Standard Metrics):**
    ```yaml
    generation:
        strategies: 
            - sample_data_path: "${hydra:runtime.cwd}/data/generation/sample_generation_data.json"
            - evaluation_metrics:
                - groundedness
                - answer_relevancy
            - g_eval_config:
                mode: "standard"
                metric_name: "Answer Relevancy"  # 선택 가능: Answer Relevancy, Consistency, Fluency, Groundness, Relevancy
                metric_llm:
                    model_name: "gpt-4"
                    temperature: 0.0
                    max_tokens: 1024
    ```
    
    **커스텀 메트릭 사용법:**
    ```yaml
    generation:
        strategies: 
            - sample_data_path: "${hydra:runtime.cwd}/data/generation/sample_generation_data.json"
            - g_eval_config:
                mode: "custom"
                metric_name: "Technical Accuracy"
                metric_description: "Evaluating how technically accurate and precise the answer is"
                metric_criterion: |
                    - 1: Very Poor. The answer contains significant technical errors.
                    - 2: Poor. The answer has some technical accuracy but contains notable errors.
                    - 3: Fair. The answer is generally accurate but lacks precision.
                    - 4: Good. The answer is technically accurate with minor issues.
                    - 5: Excellent. The answer is perfectly accurate and technically precise.
                metric_llm:
                    model_name: "gpt-4"
                    temperature: 0.0
                    max_tokens: 1024
    ```

</details>

<details>
<summary>Benchmark Module</summary>

- 핵심 기능
    - NIAH (Needle In A Haystack) 평가
        - 긴 컨텍스트 내에서 특정 정보를 찾는 능력 평가
        - 다양한 테스트 케이스 지원:
            - single_needle : 단일 정보 검색
            - multi_needle : 다중 정보 검색
            - complex_info : 복잡한 정보 검색
            - password_test : 암호 찾기 테스트
            - location_test : 위치 정보 찾기 테스트
        - 컨텍스트 길이와 깊이에 따른 성능 분석
        - 선택적 테스트 케이스 실행 지원
        - JSON/YAML 설정 파일 지원

- 사용법
    - `conf/config.yaml`의 `benchmark` 섹션에 아래 내용을 참고하여 작성한다.
    
    **기본 사용법:**
    ```yaml
    benchmark:
        strategies:
            - llm_endpoint: "openai/gpt-4o"
            - needle_config_path: "${hydra:runtime.cwd}/data/benchmark/NIAH/needle_config.json"
            - NIAH:
                context_lengths: [1000, 2000, 4000]
                document_depth_percents: [0.1, 0.5, 0.9]
                num_samples_per_test: 2
                save_results: true
                save_contexts: false
                test_cases: ["single_needle", "multi_needle", "complex_info"]  # 실행할 테스트 선택
    ```
    
    **needle_config.json 형식:**
    ```json
    {
        "single_needle": {
            "needles": ["The secret code is ALPHA-7234."],
            "question": "What is the secret code?",
            "true_answer": "ALPHA-7234"
        },
        "multi_needle": {
            "needles": [
                "The meeting will be held in Conference Room B.",
                "The meeting time is 3:30 PM.",
                "The meeting date is next Tuesday."
            ],
            "question": "When and where is the meeting?",
            "true_answer": "The meeting will be held in Conference Room B at 3:30 PM next Tuesday."
        },
        "complex_info": {
            "needles": [
                "Dr. Smith discovered the rare element Xenium in 2019.",
                "Xenium has atomic number 142.",
                "The element exhibits superconducting properties at room temperature."
            ],
            "question": "What are the key facts about Xenium?",
            "true_answer": "Dr. Smith discovered Xenium in 2019. It has atomic number 142 and exhibits superconducting properties at room temperature."
        }
    }
    ```
    
    **특정 테스트만 실행하기:**
    ```yaml
    benchmark:
        strategies:
            - llm_endpoint: "openai/gpt-4o"
            - needle_config_path: "${hydra:runtime.cwd}/data/benchmark/NIAH/needle_config.json"
            - NIAH:
                context_lengths: [1000, 2000]
                document_depth_percents: [0.1, 0.5]
                num_samples_per_test: 1
                save_results: true
                save_contexts: false
                test_cases: ["single_needle", "multi_needle"]  # 2개 테스트만 실행
    ```

    **FunctionChat Bench 평가:**
    
    FunctionChat-Bench는 한국어 LLM의 함수 호출(function calling) 능력을 평가하는 벤치마크입니다.
    
    - 주요 특징:
        - Dialog/SingleCall 두 가지 평가 타입 지원
        - Exact Match와 LLM-as-Judge 평가 방식
        - OpenAI 호환 API 지원 (로컬 모델 사용 가능)
        - 한국어 함수 호출 시나리오 평가
    
    **기본 사용법:**
    ```yaml
    function_chat:
        strategies:
            # 평가할 모델 설정
            - llm_model_name: "gpt-4o"
            - llm_api_key: "${common.OPENAI_API_KEY}"
            - llm_endpoint: "https://api.openai.com/v1"
            
            # 평가자 모델 설정 (선택사항, 기본값: GPT-4)
            - evaluator_model: "gpt-4"
            - evaluator_endpoint: "https://api.openai.com/v1"
            
            # 평가 설정
            - evaluation_types: ["dialog", "singlecall"]  # 평가 타입 선택
            - data_path: "${hydra:runtime.cwd}/data/benchmark/functionchat_bench"
            - dataset_files:  # 커스텀 데이터셋 파일 지정 (선택사항)
                dialog: "FunctionChat-Dialog-Sample.jsonl"
                singlecall: "FunctionChat-Singlecall-Sample.jsonl"
            - temperature: 0.0
            - tool_choice: "auto"
            - only_exact: false  # true: exact match만, false: LLM 평가 포함
    ```
    
    **로컬 모델 사용 예시:**
    ```yaml
    function_chat:
        strategies:
            # vLLM 또는 다른 OpenAI 호환 서버 사용
            - llm_model_name: "llama-3-70b-instruct"
            - llm_api_key: "dummy-key"  # 로컬 서버에서는 무시됨
            - llm_endpoint: "http://localhost:8000/v1"
            
            # 평가자는 GPT-4 사용 (권장)
            - evaluator_model: "gpt-4"
            - evaluator_endpoint: "https://api.openai.com/v1"
            
            - evaluation_types: ["singlecall"]
            - only_exact: false
    ```
    
    **데이터셋 형식:**
    - Dialog: 다중 턴 대화에서의 함수 호출 평가
    - SingleCall: 단일 쿼리에 대한 함수 호출 평가
    - 각 예제는 tools, query, ground_truth를 포함
    
    **평가 결과:**
    - Accuracy: 전체 정답률
    - 상세 결과는 `outputs/function_chat_summary.json`에 저장

</details>