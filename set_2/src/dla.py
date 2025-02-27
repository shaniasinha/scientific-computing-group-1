import numpy as np
import matplotlib.pyplot as plt
import os

class DiffusionLimitedAggregation:
    """
    Base class for Diffusion-Limited Aggregation (DLA).
    Supports deterministic and stochastic variants through inheritance.
    """

    def __init__(self, grid_size=100, eta=1.0, max_iterations=5000, omega=1.9, tol=1e-5, save_path="set_2/results/determinitic_dla"):
        self.N = grid_size
        self.eta = eta                                      # Growth parameter controlling cluster compactness
        self.max_iterations = max_iterations
        self.omega = omega                                  # Relaxation parameter for SOR
        self.tol = tol                                      # Tolerance for Laplace equation convergence

        # Ensure results are saved in the correct directory
        self.save_path = save_path                          # Save path for results
        os.makedirs(self.save_path, exist_ok=True)

        # Initialize grid and concentration field
        self.grid = np.zeros((self.N, self.N), dtype=int)   # 0 = empty, 1 = occupied
        self.concentration = np.ones((self.N, self.N))      # Nutrient concentration field

        # Place the initial seed at the center
        self.grid[self.N // 2, self.N // 2] = 1  

    def solve_laplace(self):
        """
        Solves the Laplace equation ∇²c = 0 using the Successive Over Relaxation (SOR) method.
        Updates self.concentration in place.
        """
        for _ in range(self.max_iterations):  # Max iterations for convergence
            prev_concentration = self.concentration.copy()
            for i in range(1, self.N - 1):
                for j in range(1, self.N - 1):
                    if self.grid[i, j] == 0:  # Only update empty sites
                        self.concentration[i, j] = (1 - self.omega) * self.concentration[i, j] + self.omega * (
                            0.25 * (self.concentration[i + 1, j] + self.concentration[i - 1, j] +
                                    self.concentration[i, j + 1] + self.concentration[i, j - 1])
                        )
            # Convergence check
            if np.max(np.abs(self.concentration - prev_concentration)) < self.tol:
                break

    def get_growth_candidates(self):
        """
        Identifies empty sites adjacent to occupied sites (growth candidates).
        Returns a list of candidate coordinates.
        """
        candidates = []
        for i in range(1, self.N - 1):
            for j in range(1, self.N - 1):
                if self.grid[i, j] == 0:  # Empty site
                    # Check if any neighbor is occupied
                    if (self.grid[i + 1, j] == 1 or self.grid[i - 1, j] == 1 or
                        self.grid[i, j + 1] == 1 or self.grid[i, j - 1] == 1):
                        candidates.append((i, j))
        return candidates

    def compute_growth_probabilities(self, candidates):
        """
        Computes growth probabilities for each candidate site based on nutrient concentration.
        Returns a normalized probability array.
        """
        probabilities = np.array([self.concentration[i, j] ** self.eta for i, j in candidates])
        return probabilities / np.sum(probabilities) if np.sum(probabilities) > 0 else probabilities
    
    def select_growth_site(self, candidates, probabilities):
        """
        Uses np.choice() to select a growth site based on probabilities.
        This method allows deterministic or stochastic selection.
        """
        if len(candidates) == 0:
            return None

        # Select a site based on probabilities using np.choice()
        chosen_index = np.random.choice(len(candidates), p=probabilities)
        return candidates[chosen_index]

    def run_simulation(self):
        """
        Runs the DLA simulation until the maximum iterations are reached.
        This method should be overridden in subclasses to implement different growth strategies.
        """
        raise NotImplementedError("Subclasses must implement `run_simulation`.")

    def visualize(self, title="Diffusion-Limited Aggregation"):
        """
        Displays the final aggregated structure.
        """
        plt.imshow(self.grid, cmap='gray')
        plt.title(title)
        plt.show()
