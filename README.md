# BERGEN-UP

>   New version of BERGEN (a.k.a BERGEN UP✨)

[**BERGEN**](https://github.com/naver/bergen?tab=readme-ov-file) (*BEnchmarking Retrieval-augmented GENeration*) is a library designed to benchmark RAG systems with a focus on question-answering (QA) by **NAVER Labs**. It addresses the challenge of inconsistent benchmarking in comparing approaches and understanding the impact of each component in a RAG pipeline. Unlike BERGEN, BERGEN-UP is an end-to-end evaluation pipeline that enhanced focuses on the diversity of RAG pipelines and the functionality of each modules.


## 🍒 Key Feature
- **E2E Evaluation Pipeline** for RAG
    - Chunking
        - token level
            - recall
            - precision
            - iou
    - Pre-Retrieval
    - Retrieval
        - evaluation level
            - precision@k
            - recall@k  
            - f1@k
            - ndcg@k
            - hit_rate@k
            - mrr
    - Post-Retrieval
    - Generation
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
