# BERGEN-UP

>   New version of BERGEN (a.k.a BERGEN UP✨)

[**BERGEN**](https://github.com/naver/bergen?tab=readme-ov-file) (*BEnchmarking Retrieval-augmented GENeration*) is a library designed to benchmark RAG systems with a focus on question-answering (QA) by **NAVER Labs**. It addresses the challenge of inconsistent benchmarking in comparing approaches and understanding the impact of each component in a RAG pipeline. Unlike BERGEN, BERGEN-UP is an end-to-end evaluation pipeline that enhanced focuses on the diversity of RAG pipelines and the functionality of each modules.


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
    - *Post-Retrieval*
    - *Generation*
        - static metric
            - groundedness
            - answer_relevancy
- **BENCHMARK Pipeline**
    - *Bench-Test*
        - BEIR
        - ASQA
        - TriviaQA
        - HotpotQA
        - WikiQA
        - NQ
- **Extra Module** for RAG
    - Generate Synthetic Dataset
        - QA (= Question Answering)


## 🥑 How to run pipeline?

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
        - Metric : 
            - groundedness : 생성된 답변이 제공된 컨텍스트에 얼마나 근거하는지 평가 (0-1)
            - answer_relevancy : 생성된 답변이 질문에 얼마나 관련성이 있는지 평가 (0-1)

- 사용법
    - `conf/config.yaml`의 `generation` 섹션에 아래 내용을 참고하여 작성한다.
    ```yaml
    generation:
        strategies: 
            - sample_data_path: "${hydra:runtime.cwd}/data/generation/sample_generation_data.json"
            - evaluation_metrics:
                - groundedness
                - answer_relevancy
    ```

</details>
