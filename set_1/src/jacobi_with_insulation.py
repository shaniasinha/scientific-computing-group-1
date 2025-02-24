from numba import njit
import numpy as np
from src.set_1.time_dependent_diffusion import TimeDependentDiffusion
from matplotlib import pyplot as plt

@njit
def numba_jacobi_insulation(c, N, max_iter, tol, object, alpha):
    """
    Perform the Jacobi iteration using Numba for speed-up, with fixed sinks.
    """
    iterations = []
    diffs = []
    c_new = c.copy()  # Temporary array for updates
    
    for iteration in range(max_iter):
        c_old = c.copy()

        # Interior points update
        for i in range(1, N-1):
            for j in range(1, N-1):
                if object[i, j] == 0:  # Only update if not in a sink/insulated region
                    c_new[i, j] = 0.25 * (c_old[i+1, j] + c_old[i-1, j] + c_old[i, j+1] + c_old[i, j-1])

            # Periodic boundary conditions
            c_new[i, 0] = 0.25 * (c_old[i+1, 0] + c_old[i-1, 0] + c_old[i, 1] + c_old[i, -1])
            c_new[i, -1] = 0.25 * (c_old[i+1, -1] + c_old[i-1, -1] + c_old[i, 0] + c_old[i, -2])

        # Keep insulation regions unchanged
        for i in range(N):
            for j in range(N):
                if object[i, j] == 1:
                    c_new[i, j] = c_old[i, j]
        
        # Swap arrays for next iteration
        c[:] = c_new

        # Convergence check
        diff = np.linalg.norm(c - c_old)

        iterations.append(iteration)
        diffs.append(diff)
        
        if diff < tol:
            return iterations, c, diffs  # Converged

    return iterations, c, diffs  # Reached max iterations

class JacobiIteration(TimeDependentDiffusion):
    def __init__(self, N, max_iter=10000, tol=1e-5, activated_objects=[]):
        super().__init__(N=N, simulation_time=1.0, fig_name="laplace_solution")
        self.max_iter = max_iter
        self.tol = tol
        self.num_steps = 1                              # No time-stepping needed
        self.object = np.zeros((N, N), dtype=np.int32)
        self.alpha = np.ones((N, N), dtype=np.float32)  # Conductivity coefficients
        self.add_insulation(activated_objects)
    
    def add_insulation(self, activated_objects=[]):
        """
        Define insulating regions where values remain unchanged.
        """
        N = self.N

        # Example: Rectangular Insulator
        for insulation in activated_objects:
            if insulation == "rectangle":
                self.object[N//2:N//2+10, N//2:N//2+15] = 1     # Block updates in this area
                self.alpha[N//2:N//2+10, N//2:N//2+15] = 0.8    # 80% conductivity
            elif insulation == "circle":
                ins_center = (N // 2, N // 2)   # Center of circle
                ins_radius = N // 6             # Radius of the insulating region

                for i in range(N):
                    for j in range(N):
                        if (i - ins_center[0])**2 + (j - ins_center[1])**2 <= ins_radius**2:
                            self.object[i, j] = 1   # Mark as insulating material
                            self.alpha[i, j] = 0.3  # 30% conductivity (low diffusion rate)

    
    def solve(self):
        """
        Solve the Laplace equation using a Numba-optimized Jacobi iteration with sinks.
        """
        iterations, c, diffs = numba_jacobi_insulation(self.c, self.N, self.max_iter, self.tol, self.object, self.alpha)
        
        if len(iterations) < self.max_iter:
            print(f"Converged after {len(iterations)} iterations")
            return iterations, c, diffs
        else:
            print("Reached maximum iterations")
            return iterations, c, diffs
    
    def plot_solution(self):
        """
        Plot the solution of the Laplace equation.
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(self.c, extent=[0, 1, 0, 1], origin='lower', cmap='hot')
        plt.colorbar(label='Concentration')
        plt.title(f"Jacobi with Insulation (N = {self.N})", fontsize=18, fontweight='bold')
        plt.xlabel('x', fontsize=16)
        plt.ylabel('y', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(f"results/set_1/numerical_methods/jacobi_with_insulation_N_{self.N}.png")
        plt.show()