import numpy as np
import matplotlib.pyplot as plt
import os
from numba import njit
from tqdm import tqdm
import matplotlib.animation as animation

@njit
def numba_sor(c, N, omega, max_iter, tol, grid):
    """
    Perform the Successive Over-Relaxation (SOR) iteration with periodic boundary conditions,
    ensuring that cluster-occupied sites remain at c = 0.
    """
    iterations = [] # Store the number of iterations for convergence

    for iteration in range(max_iter):
        c_old = c.copy()

        # Occupy sites with cluster before solving Laplace equation
        for i in range(1, N-1):
            for j in range(1, N-1):
                if grid[i, j] == 1:
                    c[i, j] = 0

        for i in range(1, N-1):
            for j in range(1, N-1):
                # Set concentration of cluster sites as 0
                if grid[i, j] == 1:  
                    c[i, j] = 0
                else:
                    c[i, j] = (1 - omega) * c[i, j] + omega * 0.25 * (
                        c[i+1, j] + c[i-1, j] + c[i, j+1] + c[i, j-1]
                    )

            # Periodic boundary conditions
            c[i, 0] = (1 - omega) * c[i, 0] + omega * 0.25 * (
                c[i+1, 0] + c[i-1, 0] + c[i, 1] + c[i, -1]
            )
            c[i, -1] = (1 - omega) * c[i, -1] + omega * 0.25 * (
                c[i+1, -1] + c[i-1, -1] + c[i, 0] + c[i, -2]
            )

        # Store the number of iterations
        iterations.append(iteration)

        # Check for convergence:
        diff = np.linalg.norm(c - c_old)
        if diff < tol:
            return iterations, c

    return iterations, c  # Reached max iterations

class DiffusionLimitedAggregation:
    """
    Class for steady-state (Laplacian growth) Diffusion-Limited Aggregation (DLA).
    Supports deterministic and stochastic variants through inheritance.
    """

    def __init__(self, grid_size=100, eta=1.0, max_iterations=5000, tol=1e-5, save_path="/Users/shaniasinha/Desktop/UvA/Academics/scientific-computing/scientific-computing-group-1/set_2/results/dla_snaps"):
        self.N = grid_size                                  
        self.eta = eta                                      # Growth parameter controlling cluster compactness
        self.max_iter = max_iterations
        self.tol = tol                                      # Tolerance for Laplace equation convergence

        # Ensure results are saved in the correct directory
        self.save_path = save_path                          # Save path for results
        os.makedirs(self.save_path, exist_ok=True)

        # Initialize grid and concentration field
        self.grid = np.zeros((self.N, self.N), dtype=np.float64)               # 0 = empty, 1 = occupied
        self.concentration = np.zeros((self.N, self.N), dtype=np.float64)      # Concentration field

        # Apply initial boundary conditions
        self.apply_boundary_conditions()

        # Place the initial seed at the center of the bottom boundary
        self.grid[0, self.N // 2] = 1  

    def apply_boundary_conditions(self):
        """
        Apply boundary conditions.
        """
        self.concentration[-1, :] = 1  # Top boundary
        self.concentration[0, :] = 0   # Bottom boundary

    def solve_sor(self, omega=1.8):
        """
        Solve the Laplace equation using Numba-optimized Successive Over-Relaxation.
        """
        iterations, _ = numba_sor(self.concentration, self.N, omega, self.max_iter, self.tol, self.grid)
        # print(f"Concentration field : {self.concentration}")
        if len(iterations) < self.max_iter:
            # print(f"Converged after {len(iterations)} iterations")
            return iterations, self.concentration
        else:
            print("Reached maximum iterations")
            return iterations, self.concentration
        return None

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
        # print(candidates)               
        return candidates

    def compute_growth_probabilities(self, candidates):
        """
        Computes growth probabilities for each candidate site based on nutrient concentration.
        Returns a normalized probability array.
        """
        probabilities = np.array([self.concentration[i, j] ** self.eta for i, j in candidates])
        # print(f"probabilities = {probabilities}")
        # print(f"sum = {np.sum(probabilities)}")
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

    def visualize(self, save_animation=True, fps=10, title="DLA Growth Over Time"):
        """
        Creates an animated visualization of the DLA process, showing both the cluster 
        and the concentration field over time.

        Parameters:
        - save_animation (bool): If True, saves the animation as a GIF.
        - fps (int): Frames per second for the animation.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # Set up the cluster plot
        ax1 = axes[0]
        cluster_plot = ax1.imshow(self.grid_history[0], cmap='binary', vmin=0, vmax=1)
        ax1.set_title("Cluster Growth")
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Set up the concentration field plot
        ax2 = axes[1]
        concentration_plot = ax2.imshow(self.concentration_history[0], cmap='hot')
        ax2.set_title("Concentration Field")
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Colorbars
        plt.colorbar(cluster_plot, ax=ax1, label='Occupied (1) / Empty (0)')
        plt.colorbar(concentration_plot, ax=ax2, label='Concentration')

        # Update function for animation
        def update(frame):
            cluster_plot.set_array(self.grid_history[frame])
            concentration_plot.set_array(self.concentration_history[frame])
            fig.suptitle(f"Frame {frame+1}/{len(self.grid_history)}", fontsize=16)

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(self.grid_history), interval=1000//fps)

        # Save animation as GIF if required
        if save_animation:
            ani.save(f"{self.save_path}/dla_growth.gif", writer='pillow', fps=fps)
        
        plt.show()


    # def plot_solution(self):
    #     """
    #     Plot the solution of the Laplace equation.
    #     """
    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(self.c, extent=[0, 1, 0, 1], origin='lower', cmap='hot')
    #     plt.colorbar(label='Concentration')
    #     plt.title('Successive Over-Relaxation Solution', fontsize=18, fontweight='bold')
    #     plt.xlabel('x', fontsize=16)
    #     plt.ylabel('y', fontsize=16)
    #     plt.xticks(fontsize=14)
    #     plt.yticks(fontsize=14)
    #     plt.show()

    # def run_simulation(self):
    #     """
    #     Runs the full DLA simulation: iteratively solving the Laplace equation and growing the cluster.
    #     """
    #     for _ in tqdm(range(self.max_iter), desc="Running DLA simulation"):
    #         # Solve Laplace equation for the concentration field using SOR
    #         _, self.concentration = self.solve_sor()

    #         # Get candidate sites for growth
    #         candidates = self.get_growth_candidates()
    #         if not candidates:
    #             break 

    #         # Compute growth probabilities
    #         probabilities = self.compute_growth_probabilities(candidates)

    #         # Select the site using np.choice() based on probabilities
    #         new_site = self.select_growth_site(candidates, probabilities)
    #         if new_site:
    #             self.grid[new_site] = 1             # Grow at the selected site
    #             self.concentration[new_site] = 0    # Enforce absorbing condition

    #             # Reset concentration field to adjust based on the cluster & solve again
    #             self.concentration = np.zeros((self.N, self.N), dtype=np.float64) 
    #             _, self.concentration = self.solve_sor()  

    #     self.visualize("Stochastic DLA with η = " + str(self.eta))

    def run_simulation(self, omega_input=1.8):
        """
        Runs the full DLA simulation, storing frames for animation.
        """
        self.grid_history = []
        self.concentration_history = []

        for step in tqdm(range(self.max_iter), desc="Running DLA simulation"):
            self.apply_boundary_conditions()
            # Solve Laplace equation for the concentration field using SOR
            _, self.concentration = self.solve_sor(omega=omega_input)
            # Prevent negative values that may arise due to numerical errors
            self.concentration = np.clip(self.concentration, 0, 1)  
            # print concentration of coordinate (1, 50)
            # print(self.concentration)

            # Store frame for visualization
            self.grid_history.append(self.grid.copy())
            self.concentration_history.append(self.concentration.copy())

            candidates = self.get_growth_candidates()
            if not candidates:
                break

            probabilities = self.compute_growth_probabilities(candidates)
            new_site = self.select_growth_site(candidates, probabilities)
            if new_site:
                self.grid[new_site] = 1
                # self.concentration[new_site] = 0 # TODO: remove, redundant
                # self.concentration = np.zeros((self.N, self.N))
                # _, _ = self.solve_sor()

        self.visualize()


