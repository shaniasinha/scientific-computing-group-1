from numba import njit
import numpy as np
from src.set_1.time_dependent_diffusion import TimeDependentDiffusion
from matplotlib import pyplot as plt

@njit
def numba_jacobi(c, N, max_iter, tol, sinks):
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
                if sinks[i, j] == 0:  # Only update if not in a sink region
                    c_new[i, j] = 0.25 * (c_old[i+1, j] + c_old[i-1, j] + c_old[i, j+1] + c_old[i, j-1])

            # Periodic boundary conditions
            c_new[i, 0] = 0.25 * (c_old[i+1, 0] + c_old[i-1, 0] + c_old[i, 1] + c_old[i, -1])
            c_new[i, -1] = 0.25 * (c_old[i+1, -1] + c_old[i-1, -1] + c_old[i, 0] + c_old[i, -2])

        # Enforce sinks with explicit loop
        for i in range(N):
            for j in range(N):
                if sinks[i, j] == 1:
                    c_new[i, j] = 0
        
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
    def __init__(self, N, max_iter=10000, tol=1e-5):
        super().__init__(N=N, simulation_time=1.0, fig_name="laplace_solution")
        self.max_iter = max_iter
        self.tol = tol
        self.num_steps = 1  # No time-stepping needed
        self.sinks = np.zeros((N, N), dtype=np.int32)  # Initialize sink array as int
        self.add_sinks()
    
    def add_sinks(self):
        """
        Define three rectangular sink regions with zero concentration.
        """
        N = self.N
        # Sink 1: Small rectangle
        # self.sinks[N//4:N//4+5, N//4:N//4+10] = 1
        # Sink 2: Medium rectangle
        # self.sinks[N//2:N//2+10, N//3:N//3+15] = 1
        # Sink 3: Large rectangle
        # self.sinks[3*N//4:3*N//4+15, N//2:N//2+20] = 1

        # Sink 4: Triangular region (Right-angled triangle)
        # tri_base_center = N // 2  # Center of the base
        # tri_base_width = 5  # Width of the base
        # tri_height = 5  # Height of the triangle
        # tri_base_start = tri_base_center - tri_base_width // 2
        # tri_apex = tri_base_start + tri_height
        
        # for i in range(tri_base_start, tri_apex):  # Triangle height
        #     left_bound = tri_base_center - (i - tri_base_start)  # Left boundary of the triangle
        #     right_bound = tri_base_center + (i - tri_base_start)  # Right boundary of the triangle
        #     self.sinks[i, left_bound:right_bound + 1] = 1  # Fill in the triangle region

        # Sink 5: Equilateral triangular sink
        # tri_apex = 2*(N // 3)               # Top of the triangle
        # tri_base_center = N // 3        # Center of the base
        # tri_height = 5                      # Triangle height
        
        # # for i in range(tri_apex, tri_apex + tri_height):        # Expanding rows downward
        # #     row_width = (i - tri_apex) * 2                      # Expands symmetrically
        # #     left_bound = tri_base_center - row_width // 2
        # #     right_bound = tri_base_center + row_width // 2
        # #     self.sinks[i, left_bound:right_bound + 1] = 1       # Fill the equilateral region
        
        # # Sink 6: Equilateral triangular sink (upward)
        # tri_base = N // 2 + tri_height      # Set base at a higher row

        # for i in range(tri_base, tri_base - tri_height, -1):    # Expanding rows upward
        #     row_width = (tri_base - i) * 2                      # Shrinks as we move up
        #     left_bound = tri_base_center - row_width // 2
        #     right_bound = tri_base_center + row_width // 2
        #     self.sinks[i, left_bound:right_bound + 1] = 1       # Fill the equilateral region

        # Sink 7: Circular sink
        circle_center = (2*(N // 3), 2*(N // 2))  # Center of the circle
        circle_radius = 5  # Radius of the circle
        for i in range(N):
            for j in range(N):
                if (i - circle_center[0])**2 + (j - circle_center[1])**2 < circle_radius**2:
                    self.sinks[i, j] = 1

    
    def solve(self):
        """
        Solve the Laplace equation using a Numba-optimized Jacobi iteration with sinks.
        """
        iterations, c, diffs = numba_jacobi(self.c, self.N, self.max_iter, self.tol, self.sinks)
        if iterations[-1] < self.max_iter:
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
        plt.title('Jacobi Iteration Solution with Sinks', fontsize=18, fontweight='bold')
        plt.xlabel('x', fontsize=16)
        plt.ylabel('y', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()