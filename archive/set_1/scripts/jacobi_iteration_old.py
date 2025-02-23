import numpy as np
import matplotlib.pyplot as plt
from src.set_1.time_dependent_diffusion import TimeDependentDiffusion

class JacobiIteration(TimeDependentDiffusion):
    def __init__(self, N, max_iter=10000, tol=1e-5):
        """
        Initialize the Jacobi iteration solver for a diffusion problem.
        Inherits from TimeDependentDiffusion.
        """
        super().__init__(N=N, simulation_time=1.0, fig_name="jacobi_solution")
        self.max_iter = max_iter
        self.tol = tol
        self.num_steps = 1  # No explicit time-stepping needed

    def apply_boundary_conditions(self):
        """
        Apply boundary conditions.

        >>> tdd = TimeDependentDiffusion(N=3)
        >>> tdd.apply_boundary_conditions()
        >>> tdd.c[0, :]  # Bottom row should be 0
        array([0., 0., 0.])
        >>> tdd.c[-1, :]  # Top row should be 1
        array([1., 1., 1.])
        """
        self.c[:, -1] = 1  # Top boundary
        self.c[:, 0] = 0   # Bottom boundary

    def solve(self):
        """
        Solve the steady-state diffusion equation using the Jacobi iteration method.
        """
        for iteration in range(self.max_iter):
            c_old = self.c.copy()
            new_c = self.c.copy()
            
            for i in range(self.N):
                for j in range(self.N):
                    if self.is_source(i, j):
                        new_c[i, j] = 1.0
                    elif self.is_sink(i, j):
                        new_c[i, j] = 0.0
                    else:
                        west =  c_old[i - 1, j] if i > 0 else c_old[self.N - 1, j]  # Periodic
                        east = c_old[i + 1, j] if i < self.N - 1 else c_old[0, j]  # Periodic
                        south = 0 if j == 0 else c_old[i, j - 1]  # Fixed boundary
                        north = 1 if j == self.N - 1 else c_old[i, j + 1]  # Fixed boundary
                        
                        new_c[i, j] = 0.25 * (west + east + south + north)

            diff = np.linalg.norm(new_c - c_old)
            self.c = new_c

            if diff < self.tol:
                print(f"Converged after {iteration} iterations")
                break
        else:
            print("Reached maximum iterations without full convergence")

    def is_source(self, i, j):
        """Define source points if needed."""
        return False

    def is_sink(self, i, j):
        """Define sink points if needed."""
        return False
    
    def plot_solution(self):
        """
        Plot the solution of the Laplace equation.
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(self.c, extent=[0, 1, 0, 1], origin='lower', cmap='hot')
        plt.colorbar(label='Concentration')
        plt.title('Solution of the Laplace Equation (Jacobi Iteration)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
