from enum import Enum
import numpy as np
from scipy.spatial.distance import euclidean, cityblock
from typing import List

class SimilarityMetric(str, Enum):
    """Supported similarity metrics."""
    COSINE = 'cosine'
    EUCLIDEAN = 'euclidean'
    MANHATTAN = 'manhattan'

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between vectors."""
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError("Zero vector encountered")
        
    similarity = float(np.dot(v1, v2) / (norm_v1 * norm_v2))
    return np.clip(similarity, -1.0, 1.0)

def euclidean_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate Euclidean-based similarity."""
    distance = euclidean(v1, v2)
    max_distance = euclidean(np.ones_like(v1), np.zeros_like(v1))
    similarity = 1 - (distance / max_distance)
    return float(np.clip(similarity, 0.0, 1.0))

def manhattan_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate Manhattan-based similarity."""
    distance = cityblock(v1, v2)
    max_distance = cityblock(np.ones_like(v1), np.zeros_like(v1))
    similarity = 1 - (distance / max_distance)
    return float(np.clip(similarity, 0.0, 1.0))