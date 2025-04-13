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
