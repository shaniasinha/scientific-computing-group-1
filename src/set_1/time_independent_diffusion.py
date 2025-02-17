from src.set_1.time_dependent_diffusion import TimeDependentDiffusion
from matplotlib import pyplot as plt
import numpy as np

class TimeIndependentDifussion(TimeDependentDiffusion):
    def __init__(self, N, max_iter=10000, tol=1e-5):
        # Initialize the parent class, setting dt and simulation_time to non-zero default values
        super().__init__(N=N, dt=1.0, simulation_time=1.0, fig_name="laplace_solution")
        
        # Override attributes not needed for the steady-state case
        self.max_iter = max_iter
        self.tol = tol
        self.num_steps = 1  # No time-stepping needed

    def is_stable(self):
        pass

    def solve(self):
        """
        Solve the Laplace equation using the Gauss-Seidel iterative method with periodic boundary conditions.
        """
        for iteration in range(self.max_iter):
            c_old = np.copy(self.c)

            # Update interior points
            for i in range(1, self.N - 1):
                for j in range(1, self.N - 1):
                    self.c[i, j] = 0.25 * (self.c[i+1, j] + self.c[i-1, j] + self.c[i, j+1] + self.c[i, j-1])

            # Apply periodic boundary conditions for left and right edges
            for i in range(1, self.N - 1):
                self.c[i, 0] = 0.25 * (self.c[i+1, 0] + self.c[i-1, 0] + self.c[i, 1] + self.c[i, self.N-1])  # Wrap around
                self.c[i, self.N-1] = 0.25 * (self.c[i+1, self.N-1] + self.c[i-1, self.N-1] + self.c[i, 0] + self.c[i, self.N-2])  # Wrap around

            # Check for convergence
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