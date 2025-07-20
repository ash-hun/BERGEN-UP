from dataclasses import dataclass

@dataclass
class Chunking:
    name: str

@dataclass
class PreRetrieval:
    name: str

@dataclass
class Retrieval:
    name: str

@dataclass
class PostRetrieval:
    name: str

@dataclass
class Generation:
    name: str

@dataclass
class Benchmark:
    name: str
    
@dataclass
class Evaluation:
    chunking: Chunking
    pre_retrieval: PreRetrieval
    retrieval: Retrieval
    post_retrieval: PostRetrieval
    generation: Generation
    benchmark: Benchmark