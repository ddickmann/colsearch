import torch
import numpy as np
from typing import Tuple, Dict

class VectorGym:
    """
    Simulated Environment for the Neural Voyager.
    Represents the Vector Index as a navigable space.
    """
    def __init__(self, index_vectors: torch.Tensor, index_meta: list, device: str = "cpu"):
        self.vectors = index_vectors.to(device)
        self.meta = index_meta
        self.device = device
        self.state = None
        
    def reset(self, query_vector: torch.Tensor) -> torch.Tensor:
        """
        Starts a new search episode.
        Args:
            query_vector: (D) - Initial user query
        Returns:
            observation: (D) - Initial state (Query itself or Null)
        """
        self.query = query_vector
        self.history = []
        return query_vector

    def step(self, action_vector: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        """
        Args:
            action_vector: (D) - Neural Voyager's navigation output
        
        Returns:
            observation: (D) - Embedding of the retrieved chunk
            reward: float - Signal
            done: bool - Terminal state
            info: dict - Metadata
        """
        # 1. Search Index (Exact Match for Simulation)
        # Cosine Similarity: (N, D) @ (D, 1) -> (N, 1)
        # Normalize action
        action = torch.nn.functional.normalize(action_vector, p=2, dim=0)
        
        scores = torch.matmul(self.vectors, action)
        best_idx = torch.argmax(scores).item()
        
        retrieved_vec = self.vectors[best_idx]
        retrieved_meta = self.meta[best_idx]
        
        # 2. Reward Calculation (Mock)
        # In real training, this uses NLI against Ground Truth
        reward = 0.1 # Sparse reward for retrieval
        
        # 3. State Update
        self.history.append(best_idx)
        
        # 4. Termination
        done = False
        if len(self.history) > 10:
            done = True
            
        return retrieved_vec, reward, done, retrieved_meta
