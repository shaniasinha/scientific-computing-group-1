from numba import njit
import numpy as np
from src.set_1.time_dependent_diffusion import TimeDependentDiffusion
from matplotlib import pyplot as plt

@njit
def numba_gauss_seidel(c, N, max_iter, tol, object):
    """
    Perform the Gauss-Seidel iteration using Numba for speed-up, with fixed sinks.
    """
    iterations = []
    diffs = []
    
    for iteration in range(max_iter):
        c_old = c.copy()

        # Interior points update
        for i in range(1, N-1):
            for j in range(1, N-1):
                if object[i, j] == 0:  # Only update if not in a sink region
                    c[i, j] = 0.25 * (c[i+1, j] + c[i-1, j] + c[i, j+1] + c[i, j-1])

            # Periodic boundary conditions
            c[i, 0] = 0.25 * (c[i+1, 0] + c[i-1, 0] + c[i, 1] + c[i, -1])
            c[i, -1] = 0.25 * (c[i+1, -1] + c[i-1, -1] + c[i, 0] + c[i, -2])
        
        # Enforce sinks
        for i in range(N):
            for j in range(N):
                if object[i, j] == 1:
                    c[i, j] = 0  # Sink values remain zero

        # Convergence check
        diff = np.linalg.norm(c - c_old)
        iterations.append(iteration)
        diffs.append(diff)
        
        if diff < tol:
            return iterations, c, diffs  # Converged

    return iterations, c, diffs  # Reached max iterations

class GaussSeidelIterationSinks(TimeDependentDiffusion):
    def __init__(self, N, max_iter=10000, tol=1e-5, activated_objects=[]):
        super().__init__(N=N, simulation_time=1.0, fig_name="gauss_seidel_solution")
        self.max_iter = max_iter
        self.tol = tol
        self.num_steps = 1  # No time-stepping needed
        self.object = np.zeros((N, N), dtype=np.int32)
        self.add_sinks(activated_objects)
    
    def add_sinks(self, activated_objects=[]):
        """
        Define three rectangular sink regions with zero concentration.

        Possible activated sinks:
        - "rectangle_small": Small rectangle
        - "rectangle_medium": Medium rectangle
        - "triangle_right": Right-angled triangle
        - "equilateral_upside_down": Equilateral triangle (grows downward)
        - "equilateral": Equilateral triangle (grows upward)
        - "circle_center": Circular sink
        - "circle_through_boundary": Circular sink (going through the periodic x boundary)
        """
        N = self.N

        # Switch-case for the activated sinks
        for sink in activated_objects:
            # Small rectangle
            if sink == "rectangle_small":
                length = 10
                width = 5
                self.object[(N-N//8):(N-N//8+width), (N-N//8):(N-N//8+length)] = 1
            # Medium rectangle
            elif sink == "rectangle_medium":
                length = 15
                width = 10
                self.object[(N-N//2):(N-N//2+width), (N-N//3):(N-N//3+length)] = 1
            # Right-angled triangle
            elif sink == "triangle_right":
                tri_base_center = N // 2
                tri_base_width = 5
                tri_height = 5
                tri_base_start = tri_base_center - tri_base_width // 2
                tri_apex = tri_base_start + tri_height  # Apex of the triangle  
                for i in range(tri_base_start, tri_apex):
                    left_bound = tri_base_center - (i - tri_base_start)
                    right_bound = tri_base_center + (i - tri_base_start)
                    self.object[i, left_bound:right_bound + 1] = 1
            # Equilateral triangle (downward)
            elif sink == "equilateral_upside_down":
                tri_apex = 2*(N // 3)
                tri_base_center = N // 3
                tri_height = 5
                for i in range(tri_apex, tri_apex + tri_height):
                    row_width = (i - tri_apex) * 2
                    left_bound = tri_base_center - row_width // 2
                    right_bound = tri_base_center + row_width // 2
                    self.object[i, left_bound:right_bound + 1] = 1
            # Equilateral triangle (upward)
            elif sink == "equilateral":
                tri_height = 10
                tri_base_center = N - N // 4
                tri_base = N // 2 + tri_height
                for i in range(tri_base, tri_base - tri_height, -1):
                    row_width = (tri_base - i) * 2
                    left_bound = tri_base_center - row_width // 2
                    right_bound = tri_base_center + row_width // 2
                    self.object[i, left_bound:right_bound + 1] = 1
            # Circular sink (at the center)
            elif sink == "circle_center":
                ins_center = (N // 2, N // 2)       # Center of circle
                ins_radius = N // 6                 # Radius of the insulating region

                for i in range(N):
                    for j in range(N):
                        if (i - ins_center[0])**2 + (j - ins_center[1])**2 <= ins_radius**2:
                            self.object[i, j] = 1   # Mark as insulating material
            # Circular sink (going through the periodic x boundary
            elif sink == "circle_through_boundary":
                circle_center = (2*(N // 3), 2*(N // 2))
                circle_radius = 5
                for i in range(N):
                    for j in range(N):
                        if (i - circle_center[0])**2 + (j - circle_center[1])**2 < circle_radius**2:
                            self.object[i, j] = 1
                        if (i - circle_center[0])**2 + (j - circle_center[1] - N)**2 < circle_radius**2:
                            self.object[i, j] = 1
    
    def solve(self):
        """
        Solve the Laplace equation using a Numba-optimized Gauss-Seidel iteration with sinks.
        """
        iterations, c, diffs = numba_gauss_seidel(self.c, self.N, self.max_iter, self.tol, self.object)
        
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
        plt.title(f"Gauss-Seidel with Sink(s) (N = {self.N})", fontsize=18, fontweight='bold')
        plt.xlabel('x', fontsize=16)
        plt.ylabel('y', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(f"results/set_1/numerical_methods/gs_with_sinks_N_{self.N}.png")
        plt.show()
