from src.set_1.time_dependent_diffusion import TimeDependentDiffusion
from matplotlib import pyplot as plt
import numpy as np

class JacobiIteration(TimeDependentDiffusion):
    def __init__(self, N, max_iter=10000, tol=1e-5):
        """
        Initialize the time-independent diffusion (Laplace solver) object.
        
        >>> tid = TimeIndependentDifussion(N=5)
        >>> tid.c.shape
        (5, 5)
        """
        super().__init__(N=N, simulation_time=1.0, fig_name="laplace_solution")
        self.max_iter = max_iter
        self.tol = tol
        self.num_steps = 1  # No time-stepping needed

    def solve(self):
        """
        Solve the Laplace equation using a fully vectorized (Jacobi) iteration
        with periodic boundary conditions on the left/right edges.

        >>> tid = TimeIndependentDifussion(N=5, max_iter=100, tol=1e-3)  # doctest: +ELLIPSIS
        >>> tid.solve()  # doctest: +SKIP
        >>> tid.c.shape
        (5, 5)
        """
        # Track iteration number and previous diff for convergence check
        iterations = []
        diffs = []
        for iteration in range(self.max_iter):
            c_old = self.c.copy()

            # Vectorized update for interior points:
            new_c = self.c.copy()  # We'll compute updates into new_c.
            new_c[1:-1, 1:-1] = 0.25 * (c_old[2:, 1:-1] + c_old[:-2, 1:-1] +
                                        c_old[1:-1, 2:] + c_old[1:-1, :-2])
            # Vectorized update for left boundary (j=0) for interior rows:
            new_c[1:-1, 0] = 0.25 * (c_old[2:, 0] + c_old[:-2, 0] +
                                    c_old[1:-1, 1] + c_old[1:-1, -1])
            # Vectorized update for right boundary (j=N-1) for interior rows:
            new_c[1:-1, -1] = 0.25 * (c_old[2:, -1] + c_old[:-2, -1] +
                                    c_old[1:-1, 0] + c_old[1:-1, -2])
            
            # Update the grid:
            self.c = new_c

            # Check for convergence:
            diff = np.linalg.norm(self.c - c_old)

            # Append iteration number and diff for convergence check
            iterations.append(iteration)
            diffs.append(diff)
            
            # Break if converged
            if diff < self.tol:
                print(f"Converged after {iteration} iterations")
                return iterations, self.c, diffs
        else:
            print("Reached maximum iterations")
            return iterations, self.c, diffs


    def plot_solution(self):
        """
        Plot the solution of the Laplace equation.
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(self.c, extent=[0, 1, 0, 1], origin='lower', cmap='hot')
        plt.colorbar(label='Concentration')
        plt.title('Jacobi Iteration Solution', fontsize=18, fontweight='bold')
        plt.xlabel('x', fontsize=16)
        plt.ylabel('y', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()
        
