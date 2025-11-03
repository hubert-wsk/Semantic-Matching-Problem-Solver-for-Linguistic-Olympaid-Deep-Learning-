import fasttext as ft
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple, Iterator

# fuck FastText's "Unknown"T

class exFTModel:
    def __init__(self, ft_model_path, _words_retain: int = 10_000):
        self.mdl = ft.load_model(ft_model_path)
        self.words: List[str] = self.mdl.get_words()[: _words_retain]    # fuck this; (...?)
        self._build_index()
    
    def get_knn_of_word_(self, word: str, k: int = 10) -> List[Tuple[str, float]]:
        return self.mdl.get_nearest_neighbors(word, k = k)
    
    def get_word_vec(self, word: str):
        return self.mdl.get_word_vector(word)
    
    def _build_index(self):
        self.vectors = np.stack( [self.get_word_vec(i) for i in self.words] )
        self.knn = NearestNeighbors(metric = "cosine", ).fit(self.vectors)    # n_neighbors: ...
    
    def search_knn_of_vec_(self, vec: np.ndarray, k: int = 7) -> Iterator[Tuple[str, float]]:
        vec = vec.reshape(1, -1)  # ...
        dis_, idx_ = self.knn.kneighbors(vec, n_neighbors = k)
        for d, i in zip(dis_[0], idx_[0]):
            yield self.words[i], 1 - d