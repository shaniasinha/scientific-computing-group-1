from numba import njit
from src.set_1.time_dependent_diffusion import TimeDependentDiffusion
from matplotlib import pyplot as plt
import numpy as np 

@njit
def numba_sor(c, N, omega, max_iter, tol):
    """
    Perform the Successive Over-Relaxation (SOR) iteration with periodic boundary conditions,
    optimized with Numba.
    """
    # Track iteration number and previous diff for convergence check
    iterations = []
    diffs = []
    for iteration in range(max_iter):
        c_old = c.copy()

        for i in range(1, N-1):
            for j in range(1, N-1):
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

        # Check for convergence:
        diff = np.linalg.norm(c - c_old)

        # Append iteration number and diff for convergence check
        iterations.append(iteration)
        diffs.append(diff)

        # Break if converged
        if diff < tol:
            return iterations, c, diffs  # Converged

    return iterations, c, diffs  # Reached max iterations

class SORIteration(TimeDependentDiffusion):
    def __init__(self, N, max_iter=10000, tol=1e-5):
        """
        Initialize the time-independent diffusion (Laplace solver) object.
        
        >>> tid = TimeIndependentDifussion(N=5)
        >>> tid.c.shape
        (5, 5)
        """
        # TODO: Change doctest to reflect the new class name
        super().__init__(N=N, simulation_time=1.0, fig_name="sor_solution")
        self.max_iter = max_iter
        self.tol = tol
        self.num_steps = 1  # No time-stepping needed

    def solve(self, omega=1.8):
        """
        Solve the Laplace equation using Numba-optimized Successive Over-Relaxation.
        """
        iterations, _, diffs = numba_sor(self.c, self.N, omega, self.max_iter, self.tol)
        if len(iterations) < self.max_iter:
            # print(f"Converged after {len(iterations)} iterations")
            return iterations, self.c, diffs
        else:
            print("Reached maximum iterations")
            return iterations, self.c, diffs
        return None


    def plot_solution(self):
        """
        Plot the solution of the Laplace equation.
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(self.c, extent=[0, 1, 0, 1], origin='lower', cmap='hot')
        plt.colorbar(label='Concentration')
        plt.title('Successive Over-Relaxation Solution', fontsize=18, fontweight='bold')
        plt.xlabel('x', fontsize=16)
        plt.ylabel('y', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()