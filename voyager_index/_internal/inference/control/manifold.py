import torch


class ManifoldTangentConstraint:
    """
    Constrains exploration to the local tangent plane of the data manifold.
    Prevents the Voyager from drifting into empty void space during repulsion.
    """
    def __init__(self, k_neighbors: int = 10):
        self.k_neighbors = k_neighbors

    def compute_tangent_basis(self, center: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        """
        Estimates the local tangent plane using PCA on neighbors.

        Args:
            center: (D) - The current point on the manifold
            neighbors: (K, D) - The K nearest neighbors

        Returns:
            basis: (D, D) - Eigenvectors representing local curvature
        """
        # Center the neighbors
        # neighbors: (K, D)
        centered = neighbors - center.unsqueeze(0)

        # PCA / SVD
        # U, S, V = torch.svd(centered)
        # We want the principal components (V)
        # SVD on (K, D) -> V is (D, D)
        # First few components represent the tangent plane

        try:
            _, _, V = torch.linalg.svd(centered, full_matrices=False)
        except RuntimeError:
            # Fallback for stability if singular
            return torch.eye(center.shape[0], device=center.device)

        return V

    def project_perturbation(self, perturbation: torch.Tensor, basis: torch.Tensor, top_k_components: int = 5) -> torch.Tensor:
        """
        Projects a random perturbation onto the top-k principal components.

        Args:
            perturbation: (D) - Random noise or gradient step
            basis: (D, D) - PCA basis (V from SVD)
            top_k_components: Number of dimensions to keep

        Returns:
            projected: (D) - The perturbation constrained to the manifold
        """
        # basis[0:top_k]: (k, D)
        # We want to project p onto the subspace spanned by these rows

        selected_basis = basis[:top_k_components] # (k, D)

        # Coeffs = p @ B.T -> (D) @ (D, k) -> (k)
        coeffs = torch.matmul(perturbation, selected_basis.T)

        # Reconstruction = Coeffs @ B -> (k) @ (k, D) -> (D)
        projected = torch.matmul(coeffs, selected_basis)

        return projected
