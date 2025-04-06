import re
import numpy as np
from tqdm import tqdm
from modules.chunking.chunker.fixed_token_chunker import TextSplitter
from typing import Any, List, Optional, Tuple, Union, cast

Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]

class AbstractChunker(TextSplitter):
    def __init__(
        self,
        embeddings: Optional[object],
        buffer_size: int = 3,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: Optional[float] = 95,
        sentence_split_regex: str = r"(?<=[.?!])\s+",
        **kwargs
    ):
        super().__init__(chunk_size=4000, chunk_overlap=200,**kwargs)
        self.buffer_size = buffer_size
        self.embeddings = embeddings
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.sentence_split_regex = sentence_split_regex
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
    
    def __cosine_similarity(self, X: Matrix, Y: Matrix) -> np.ndarray:
        """Row-wise cosine similarity between two equal-width matrices."""
        if len(X) == 0 or len(Y) == 0:
            return np.array([])

        X = np.array(X)
        Y = np.array(Y)
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"Number of columns in X and Y must be the same. X has shape {X.shape} "
                f"and Y has shape {Y.shape}."
            )
        X_norm = np.linalg.norm(X, axis=1)
        Y_norm = np.linalg.norm(Y, axis=1)
        # Ignore divide by zero errors run time warnings as those are handled below.
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return similarity
    
    def _combine_sentences(self, sentences: List[dict], buffer_size: int = 1) -> List[dict]:
        """Combine sentences based on buffer size."""

        for i in range(len(sentences)):
            combined_sentence = ""

            for j in range(i - buffer_size, i):
                # Check if the index j is not negative (to avoid index out of range like on the first one)
                if j >= 0:
                    # Add the sentence at index j to the combined_sentence string
                    combined_sentence += sentences[j]["sentence"] + " "

            # Add the current sentence
            combined_sentence += sentences[i]["sentence"]

            # Add sentences after the current one, based on the buffer size
            for j in range(i + 1, i + 1 + buffer_size):
                # Check if the index j is within the range of the sentences list
                if j < len(sentences):
                    # Add the sentence at index j to the combined_sentence string
                    combined_sentence += " " + sentences[j]["sentence"]

            # Then add the whole thing to your dict
            # Store the combined sentence in the current sentence dict
            sentences[i]["combined_sentence"] = combined_sentence

        return sentences

    def _calculate_cosine_distances(self, sentences: List[dict]) -> Tuple[List[float], List[dict]]:
        """Calculate cosine distances between sentences."""
        
        distances = []
        for i in tqdm(range(len(sentences) - 1), desc="Calculating cosine distances..."):
            embedding_current = sentences[i]["combined_sentence_embedding"]
            embedding_next = sentences[i + 1]["combined_sentence_embedding"]

            similarity = self.__cosine_similarity([embedding_current], [embedding_next])[0][0]

            distance = 1 - similarity

            distances.append(distance)

            sentences[i]["distance_to_next"] = distance

        return distances, sentences

    def _calculate_breakpoint_threshold(self, distances: List[float]) -> float:
        if self.breakpoint_threshold_type == "percentile":
            return cast(
                float,
                np.percentile(distances, self.breakpoint_threshold_amount),
            )
        else:
            raise ValueError(
                f"Got unexpected `breakpoint_threshold_type`: "
                f"{self.breakpoint_threshold_type}"
            )

    def _calculate_sentence_distances(self, single_sentences_list: List[str]) -> Tuple[List[float], List[dict]]:
        """Split text into multiple components."""

        _sentences = [
            {"sentence": x, "index": i} for i, x in enumerate(single_sentences_list)
        ]

        sentences = self._combine_sentences(_sentences, self.buffer_size)
        
        embeddings = [self.embeddings(x["combined_sentence"]) for x in tqdm(sentences, desc="Embedding...")]
        
        for i, sentence in enumerate(sentences):
            sentence["combined_sentence_embedding"] = embeddings[i]

        return self._calculate_cosine_distances(sentences)

    def _semantic_split_text(self, text: str) -> List[str]:
        single_sentences_list = re.split(self.sentence_split_regex, text)

        if len(single_sentences_list) == 1:
            return single_sentences_list

        distances, sentences = self._calculate_sentence_distances(single_sentences_list)

        breakpoint_distance_threshold = self._calculate_breakpoint_threshold(distances)

        indices_above_thresh = [
            i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
        ]

        chunks = []
        start_index = 0

        # Iterate through the breakpoints to slice the sentences
        for index in tqdm(indices_above_thresh, desc="Create Semantic Chunks..."):
            # The end index is the current breakpoint
            end_index = index

            # Slice the sentence_dicts from the current start index to the end index
            group = sentences[start_index : end_index + 1]
            combined_text = " ".join([d["sentence"] for d in group])
            chunks.append(combined_text)

            # Update the start index for the next group
            start_index = index + 1

        # The last group, if any sentences remain
        if start_index < len(sentences):
            combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])
            chunks.append(combined_text)
        return chunks

class SemanticChunker(AbstractChunker):
    """
    Splitting text by Embedding Model.
    """

    def __init__(
        self,
        embedding_model: Optional[object]=None,
        separators: Optional[List[str]]=None,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(embeddings=embedding_model, **kwargs)
        self._separators = ["\n\n", "\n", ".", "?", "!", " ", ""] if separators is None else separators

    def _split_text(self, text: str) -> List[str]:
        """ Create chunk list from a texts : Semantic Chunking """
        corpora = [text]
        documents = []
        for text in tqdm(corpora, desc="Semantic Chunking..."):
            for chunk in self._semantic_split_text(text):
                documents.append(chunk)
        return documents

    def split_text(self, text: str) -> List[str]:
        return self._split_text(text)