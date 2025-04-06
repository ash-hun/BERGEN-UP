# BERGEN-UP

>   New version of BERGEN (a.k.a BERGEN UP✨)

[**BERGEN**](https://github.com/naver/bergen?tab=readme-ov-file) (*BEnchmarking Retrieval-augmented GENeration*) is a library designed to benchmark RAG systems with a focus on question-answering (QA) by **NAVER Labs**. It addresses the challenge of inconsistent benchmarking in comparing approaches and understanding the impact of each component in a RAG pipeline. Unlike BERGEN, BERGEN-UP is an end-to-end evaluation pipeline that enhanced focuses on the diversity of RAG pipelines and the functionality of each modules.


## 🍒 Key Points
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


## 🥑 How to run pipeline?

##### 1. Write your evaluation in `conf/config.yaml`

##### 2. Run only below script
```bash
$ uv run pipeline.py label='__experiments_name__'
```
