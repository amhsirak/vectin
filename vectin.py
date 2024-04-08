import os
import shutil
import joblib
from joblib import Memory
import numpy as np


class Vectin:
    """
    Vectin is a class for managing and processing vectors.

    Args:
        name (str): Name of the Vectin instance.
        max_tokens (int): Maximum size of the array.
        cache_data (bool): Whether to cache data.
        persist_data (bool): Whether to persist data.
        data_storage_path (str): Path for storing data.
        cache_path (str): Path for caching data.

    Attributes:
        _DATA_STORAGE_PATH (str): Default path for storing data.
        _CACHE_PATH (str): Default path for caching data.
        _DATA_STORAGE_INDEX (str): Name of the index data file.
        _DATA_STORAGE_DATA (str): Name of the vector data file.
    """

    _DATA_STORAGE_PATH = "./vectdb"
    _CACHE_PATH = "./vectdb/cache"
    _DATA_STORAGE_INDEX = "vecti"
    _DATA_STORAGE_DATA = "vectd"

    def __init__(
        self,
        name: str = None,
        max_tokens: int = 768,
        cache_data: bool = True,
        persist_data: bool = False,
        data_storage_path: str = None,
        cache_path: str = None,
    ):
        self._name = name
        self._vector_data = {}
        self._vector_index = {}
        self._vocabulary = set()
        self._word_to_index = {}
        self._max_tokens = max_tokens
        self._persist_data = persist_data
        self._data_storage_path = data_storage_path

        self._init_persistence(data_storage_path, cache_path)

        if cache_data:
            if cache_path is None:
                self._cache_path = self._CACHE_PATH
            os.makedirs(self._cache_path, exist_ok=True)
            self._memory = Memory(self._CACHE_PATH, mmap_mode='r')

    def _init_persistence(self, data_storage_path: str = None, cache_path: str = None):
        if data_storage_path is None:
            self._data_storage_path = str(self._DATA_STORAGE_PATH)
        if cache_path is None:
            self._cache_path = self._CACHE_PATH
        try:
            os.makedirs(self._data_storage_path, exist_ok=True)
            os.makedirs(self._cache_path, exist_ok=True)
            print(
                f"Vectin data directory created at {self._data_storage_path}")
            print(f"Vectin cache directory created at {self._cache_path}")
        except OSError as error:
            print("Vectin data directory can not be created")

    def add_vector(self, vector_id, vector):
        """
        Add a vector to the Vectin instance.

        Args:
            vector_id: Identifier for the vector.
            vector: The vector data.
        """
        self._vector_data[vector_id] = vector
        self._update_index(vector_id, vector)

    def get_vector(self, vector_id):
        """
        Retrieve a vector from the Vectin instance.

        Args:
            vector_id: Identifier for the vector.

        Returns:
            The vector data corresponding to the given identifier.
        """
        return self._vector_data.get(vector_id)
    
    def _cosine_similarity(self, vector1, vector2):
        """
        Calculate the cosine similarity between two vectors.

        Args:
            vector1: First vector.
            vector2: Second vector.

        Returns:
            The cosine similarity between the two vectors.
        """
        dot_product = np.dot(vector1, vector2)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)
        return dot_product / (norm_vector1 * norm_vector2)

    def _update_index(self, vector_id, vector):
        """
        Update the index with a new vector.

        Args:
            vector_id: Identifier for the vector.
            vector: The vector data.
        """
        for existing_id, existing_vector in self._vector_data.items():
            similarity = self._cosine_similarity(vector, existing_vector)
            if existing_id not in self._vector_index:
                self._vector_index[existing_id] = {}
            self._vector_index[existing_id][vector_id] = similarity

    def find_similar_vectors(self, query_vector, num_results=5):
        """
        Find similar vectors to a given query vector.

        Args:
            query_vector: The vector to compare against.
            num_results: Number of similar vectors to retrieve.

        Returns:
            List of similar vectors.
        """
        results = []
        for vector_id, vector in self._vector_data.items():
            similarity = self._cosine_similarity(query_vector, vector)
            results.append((vector_id, similarity))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:num_results]

    def text_to_vector(self, sentence: str):
        """
        Convert text sentence to vector representation.

        Args:
            sentence: The text sentence to convert.

        Returns:
            The vector representation of the sentence.
        """
        self._update_vocabulary_and_index([sentence])
        tokens = sentence.lower().split()
        vector = np.zeros(self._max_tokens)
        for token in tokens:
            vector[self._word_to_index[token]] += 1
        return vector

    @staticmethod
    def sentence_word_splitter(num_of_words: int, sentence: str) -> list:
        """
        Split a sentence into chunks of words.

        Args:
            num_of_words: Number of words per chunk.
            sentence: The input sentence.

        Returns:
            List of sentence chunks.
        """
        pieces = sentence.split()
        return [" ".join(pieces[i:i+num_of_words]) for i in range(0, len(pieces), num_of_words)]

    @staticmethod
    def chunk_text_to_fixed_length(text: str, length: int):
        """
        Chunk a text into fixed-length chunks.

        Args:
            text: The input text.
            length: The desired length of each chunk.

        Returns:
            List of fixed-length text chunks.
        """
        text = text.strip()
        return [text[0+i:length+i] for i in range(0, len(text), length)]

    def chunk_sentences_to_fixed_length(self, sentences: list, max_length: int = 768):
        """
        Chunk sentences into fixed-length segments.

        Args:
            sentences: List of sentences.
            max_length: Maximum length of each segment.

        Returns:
            List of fixed-length sentence segments.
        """
        fixed_size_sentences = []
        for sentence in sentences:
            chunks = Vectin.chunk_text_to_fixed_length(
                text=sentence, length=max_length)
            fixed_size_sentences.extend(chunks)
        return fixed_size_sentences

    def chunk_sentences_to_max_tokens(self, sentences: list, max_tokens: int = 768):
        """
        Chunk sentences into segments with a maximum number of tokens.

        Args:
            sentences: List of sentences.
            max_tokens: Maximum number of tokens per segment.

        Returns:
            List of sentence segments.
        """
        fixed_size_sentences = []
        for sentence in sentences:
            tokens = sentence.lower().split()
            if len(tokens) > max_tokens:
                chunks = Vectin.sentence_word_splitter(
                    num_of_words=max_tokens, sentence=sentence)
                fixed_size_sentences.extend(chunks)
            else:
                fixed_size_sentences.append(sentence)
        return fixed_size_sentences
    
    def sentences_to_vector(self, sentences: list) -> list:
        """
        Convert list of sentences to vector representations.

        Args:
            sentences: List of sentences.

        Returns:
            Dictionary mapping sentences to their vector representations.
        """
        sentence_vectors = {}
        for sentence in sentences:
            vector = self.text_to_vector(sentence)
            sentence_vectors[sentence] = vector
        return sentence_vectors

    def _update_vocabulary_and_index(self, sentences: list):
        """
        Update vocabulary and index based on input sentences.

        Args:
            sentences: List of sentences.
        """
        for sentence in sentences:
            tokens = sentence.lower().split()
            self._vocabulary.update(tokens)
        self._word_to_index = {word: i for i,
                               word in enumerate(self._vocabulary)}

    def encode(self, sentences: list):
        """
        Encode list of sentences into vector representations.

        Args:
            sentences: List of sentences.

        Returns:
            Dictionary mapping sentences to their vector representations.
        """
        self._update_vocabulary_and_index(sentences)
        sentence_vectors = self.sentences_to_vector(sentences)
        return sentence_vectors


