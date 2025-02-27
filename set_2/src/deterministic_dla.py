from src.dla import DiffusionLimitedAggregation
import numpy as np

class DeterministicDLA(DiffusionLimitedAggregation):
    """
    Deterministic DLA class that inherits from the base DLA class.
    Growth occurs at the site with the highest probability.
    """

    def run_simulation(self):
        for _ in range(self.max_iterations):
            # Solve Laplace equation for the nutrient concentration field
            self.solve_laplace()

            # Get candidate sites for growth
            candidates = self.get_growth_candidates()
            if not candidates:
                break  # Stop if no candidates left

            # Compute growth probabilities
            probabilities = self.compute_growth_probabilities(candidates)

            # Select the site using np.choice() based on probabilities
            new_site = self.select_growth_site(candidates, probabilities)
            if new_site:
                self.grid[new_site] = 1  # Grow at the selected site

        self.visualize("Deterministic DLA")


