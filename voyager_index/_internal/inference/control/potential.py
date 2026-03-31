import torch
import torch.nn as nn
import torch.nn.functional as F

class TabuPotentialField:
    """
    Implements the Neuro-Symbolic Control Law.
    Minimizes E(v) to find the optimal search vector under constraints.
    """
    def __init__(self, lambda_repulsion: float = 1.0, trust_region: float = 0.1, lr: float = 0.1, steps: int = 5):
        self.lambda_repulsion = lambda_repulsion
        self.trust_region = trust_region
        self.lr = lr
        self.steps = steps

    def solve(self, 
              v_prop: torch.Tensor, 
              memory_vectors: torch.Tensor, 
              memory_intents: torch.Tensor, 
              current_intent: torch.Tensor,
              sigmas: torch.Tensor
             ) -> torch.Tensor:
        """
        Args:
            v_prop: (B, D) - Proposed vector from Neural Core
            memory_vectors: (B, M, D) - Visited vectors
            memory_intents: (B, M) - Discrete intent IDs of visits
            current_intent: (B) - Current intent ID
            sigmas: (B, M) - Local density of visited nodes
            
        Returns:
            v_opt: (B, D) - Optimized query vector
        """
        B, D = v_prop.shape
        # Initialize v with proposal
        v = v_prop.clone().detach().requires_grad_(True)
        
        # Optimizer (SGD or simple gradient descent manually)
        # We do manual updates for control
        
        for _ in range(self.steps):
            # Compute Energy
            
            # 1. Fidelity Term: 1 - <v, v_prop> (Minimize angle)
            # Use cosine distance approximation on normalized vectors
            # Assuming inputs already normalized, but we enforce norm(v)=1
            v_norm = F.normalize(v, p=2, dim=1)
            fidelity = 1.0 - torch.sum(v_norm * v_prop, dim=1) # (B)
            
            # 2. Repulsion Term
            if memory_vectors is not None and memory_vectors.shape[1] > 0:
                # Mask: I(s == s_i)
                # memory_intents: (B, M), current_intent: (B) -> (B, 1)
                mask = (memory_intents == current_intent.unsqueeze(1)).float()
                
                # Cosine Dist to memories: 1 - <v, v_i>
                # v: (B, D) -> (B, 1, D)
                # mem: (B, M, D)
                dists = 1.0 - torch.sum(v_norm.unsqueeze(1) * memory_vectors, dim=2) # (B, M)
                
                # Gaussian Repulsion: exp( - dist / (2*sigma^2) )
                # sigmas: (B, M)
                # epsilon to avoid div zero
                repulsion = torch.exp(-dists / (2 * sigmas.pow(2) + 1e-6))
                
                # Weighted Sum
                repulsion_energy = torch.sum(mask * repulsion, dim=1) # (B)
            else:
                repulsion_energy = torch.zeros(B, device=v.device)
            
            total_energy = fidelity + self.lambda_repulsion * repulsion_energy
            
            # Gradient Step
            grad = torch.autograd.grad(total_energy.sum(), v)[0]
            
            # Trust Region Constraint (Clip Norm of Grad * LR)
            # Update: v_new = v - lr * grad
            # Note: We update the un-normalized 'v', then normalize
            
            with torch.no_grad():
                update = -self.lr * grad
                
                # Trust Region Clipping (Simple Magnitude Clip)
                update_norm = update.norm(p=2, dim=1, keepdim=True)
                scale = torch.clamp(update_norm, max=self.trust_region) / (update_norm + 1e-8)
                update = update * scale
                
                v.add_(update)
                
                # Manifold Constraint: Project back to Unit Hypersphere
                v.div_(v.norm(p=2, dim=1, keepdim=True) + 1e-8)
                
            # Re-enable grad for next step
            v.requires_grad_(True)
            
        return v.detach()
