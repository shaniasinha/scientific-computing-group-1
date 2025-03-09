import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import os
from tqdm import tqdm

@njit
def numba_sor(c, mask, N, omega, max_iter, tol):
    """
    Perform the Successive Over-Relaxation (SOR) iteration with appropriate boundary conditions.
    """
    for iteration in range(max_iter):
        c_old = c.copy()
        
        # Update interior points only
        for i in range(1, N-1):
            for j in range(1, N-1):
                # Skip update if this point is part of the cluster
                if mask[i, j]:
                    continue
                
                c[i, j] = (1 - omega) * c[i, j] + omega * 0.25 * (
                    c[i+1, j] + c[i-1, j] + c[i, j+1] + c[i, j-1]
                )
        
        # Check for convergence
        diff = np.linalg.norm(c - c_old)
        if diff < tol:
            return c, iteration
    
    return c, max_iter

class DLASimulation:
    def __init__(self, N=100, max_iter_sor=1000, tol=1e-5, omega=1.8, eta=1.0, output_dir="results/dla_output_eta1.0"):
        """
        Initialize the DLA simulation.
        """
        self.N = N
        self.max_iter_sor = max_iter_sor
        self.tol = tol
        self.omega = omega
        self.eta = eta
        
        # Grid spacing
        self.dx = 1.0 / (N - 1)
        
        # Create coordinate arrays
        self.x = np.linspace(0, 1, N)
        self.y = np.linspace(0, 1, N)
        
        # Concentration field
        self.c = np.zeros((N, N), dtype=np.float64)
        
        # Mask to track aggregated particles (True where particles exist)
        self.mask = np.zeros((N, N), dtype=bool)
        
        # Initialize the seed particle at bottom center (0.5, 0)
        seed_j = N // 2  # middle column (x = 0.5)
        seed_i = 0       # bottom row (y = 0)
        self.mask[seed_i, seed_j] = True
        
        # Particle count
        self.particle_count = 1
        
        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            # Create the directory if it does not exist
            os.makedirs(output_dir)
        else:
            # Empty the directory if it exists and has files
            files = os.listdir(output_dir)
            if files:
                for file in files:
                    file_path = os.path.join(output_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
        
        # Apply initial boundary conditions
        self.apply_boundary_conditions()
    
    def apply_boundary_conditions(self):
        """
        Apply boundary conditions for the DLA simulation.
        - Top boundary: c = 1 (source)
        - Bottom and sides: no-flux (Neumann) boundaries
        - Cluster: c = 0 (absorbing/Dirichlet boundary)
        """
        # Reset the concentration field
        self.c = np.zeros((self.N, self.N), dtype=np.float64)
        
        # Top boundary (source)
        self.c[-1, :] = 1.0
        
        # Set concentration to 0 at cluster sites (absorbing boundary)
        self.c[self.mask] = 0.0
    
    def solve_laplace_sor(self):
        """
        Solve the Laplace equation using SOR method.
        """
        self.c, iterations = numba_sor(self.c, self.mask, self.N, 
                                       self.omega, self.max_iter_sor, self.tol)
        return iterations
    
    def find_growth_candidates(self):
        """
        Find all growth candidate sites (empty cells adjacent to the cluster).
        """
        candidates = []
        
        # Check all grid points
        for i in range(self.N):
            for j in range(self.N):
                # Skip if this point is already part of the cluster
                if self.mask[i, j]:
                    continue
                
                # Check if this point is a neighbor of the cluster
                # (north, east, south, west neighbors only)
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.N and 0 <= nj < self.N and self.mask[ni, nj]:
                        candidates.append((i, j))
                        break
        
        return candidates
    
    def calculate_growth_probabilities(self, candidates):
        """
        Calculate growth probabilities for all candidate sites.
        """
        # Extract concentration values at candidate sites
        concentrations = np.array([self.c[i, j] for i, j in candidates])
        
        # Apply eta parameter as exponent (p_g ~ c^eta)
        if self.eta != 1.0:
            concentrations = concentrations ** self.eta
        
        # Normalize probabilities
        probabilities = concentrations / np.sum(concentrations) if np.sum(concentrations) > 0 else np.ones_like(concentrations) / len(concentrations)
        
        return probabilities

    def grow_cluster(self):
        """
        Grow the cluster by adding a single particle based on growth probabilities.
        """
        # Find growth candidates
        candidates = self.find_growth_candidates()
        
        if not candidates:
            print("No available growth candidates")
            return False, 0
        
        # Calculate growth probabilities
        probabilities = self.calculate_growth_probabilities(candidates)
        
        # Choose a candidate based on probabilities
        idx = np.random.choice(len(candidates), p=probabilities)
        new_i, new_j = candidates[idx]
        
        # Add the chosen candidate to the cluster
        self.mask[new_i, new_j] = True
        self.particle_count += 1
        
        # Check if a particle has reached the top boundary
        if new_i == self.N - 1:  # Top row index is N-1
            print("Cluster has reached the top boundary! Stopping simulation.")
            return True, 1
        
        return True, 0
    
    def plot_domain(self, save=True):
        """
        Plot the current state of the DLA simulation.
        """
        plt.figure(figsize=(10, 10))
        
        # Create a visualization showing both concentration field and cluster
        vis = self.c.copy()
        # plt.imshow(vis, origin='lower', extent=[0, 1, 0, 1], cmap='hot', vmin=self.c, vmax=1)
        plt.imshow(self.c, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
        # plt.colorbar(label='Concentration')
        
        # Overlay cluster particles
        cluster_i, cluster_j = np.where(self.mask)
        y_coords = cluster_i * self.dx
        x_coords = cluster_j * self.dx
        plt.scatter(x_coords, y_coords, c='white', edgecolors='black', s=50)
        
        plt.colorbar(label='Concentration')
        plt.title(f'DLA Simulation - {self.particle_count} Particles (η={self.eta})', fontsize=16)
        plt.xlabel('x', fontsize=14)
        plt.ylabel('y', fontsize=14)
        
        if save:
            plt.savefig(f"{self.output_dir}/dla_eta{self.eta:.1f}_{self.particle_count:04d}.png", dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def run_simulation(self, num_particles=1000):
        """
        Run the DLA simulation until a target number of particles is reached
        or the cluster reaches the top boundary.
        """
        # Plot initial state
        self.plot_domain()
        
        # Main simulation loop
        with tqdm(total=num_particles-1) as pbar:
            while self.particle_count < num_particles:
                # Solve Laplace equation with current boundaries
                self.apply_boundary_conditions()
                self.solve_laplace_sor()
                
                # Add a new particle
                success, status = self.grow_cluster()
                if not success:
                    print("Unable to add more particles. Simulation stopped.")
                    break
                
                # Check if the cluster reached the top boundary
                if status == 1:
                    print(f"Simulation stopped after {self.particle_count} particles due to reaching top boundary")
                    # Plot final state before exiting
                    self.plot_domain()
                    break
                
                # Plot and save the current state
                self.plot_domain()
                pbar.update(1)
        
        print(f"Simulation completed with {self.particle_count} particles")