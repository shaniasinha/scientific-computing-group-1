from src.set_1.time_dependent_diffusion import TimeDependentDiffusion
from matplotlib import pyplot as plt
import numpy as np

class GaussSeidelIteration(TimeDependentDiffusion):
    def __init__(self, N, max_iter=10000, tol=1e-5):
        """
        Initialize the time-independent diffusion (Laplace solver) object.
        
        >>> tid = TimeIndependentDifussion(N=5)
        >>> tid.c.shape
        (5, 5)
        """
        #TODO: Change doctest to reflect the new class name
        super().__init__(N=N, simulation_time=1.0, fig_name="gauss_seidel_solution")
        self.max_iter = max_iter
        self.tol = tol
        self.num_steps = 1  # No time-stepping needed

    def solve(self):
        """
        Solve the Laplace equation using a fully vectorized (Gauss-Seidel) iteration
        with periodic boundary conditions on the left/right edges.

        >>> tid = TimeIndependentDifussion(N=5, max_iter=100, tol=1e-3)  # doctest: +ELLIPSIS
        >>> tid.solve()  # doctest: +SKIP
        >>> tid.c.shape
        (5, 5)
        """
        for iteration in range(self.max_iter):
            c_old = self.c.copy()

            # Vectorized update for interior points:
            self.c[1:-1, 1:-1] = 0.25 * (self.c[2:, 1:-1] + self.c[:-2, 1:-1] +
                                        self.c[1:-1, 2:] + self.c[1:-1, :-2])
            # Vectorized update for left boundary (j=0) for interior rows:
            self.c[1:-1, 0] = 0.25 * (self.c[2:, 0] + self.c[:-2, 0] +
                                    self.c[1:-1, 1] + self.c[1:-1, -1])
            # Vectorized update for right boundary (j=N-1) for interior rows:
            self.c[1:-1, -1] = 0.25 * (self.c[2:, -1] + self.c[:-2, -1] +
                                    self.c[1:-1, 0] + self.c[1:-1, -2])

            # (If needed, one could also update top and bottom boundaries.)

            # Check for convergence:
            diff = np.linalg.norm(self.c - c_old)
            if diff < self.tol:
                print(f"Converged after {iteration} iterations")
                break
        else:
            print("Reached maximum iterations")


    def plot_solution(self):
        """
        Plot the solution of the Laplace equation.
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(self.c, extent=[0, 1, 0, 1], origin='lower', cmap='hot')
        plt.colorbar(label='Concentration')
        plt.title('Solution of the Laplace Equation')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()