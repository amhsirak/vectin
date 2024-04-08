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
