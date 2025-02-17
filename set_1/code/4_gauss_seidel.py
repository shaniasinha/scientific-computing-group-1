import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 50  # Grid size (N x N)
tol = 1e-5  # Convergence tolerance
max_iter = 10000  # Maximum number of iterations

# Initialize concentration field
c = np.zeros((N, N))

# Apply boundary conditions
c[:, -1] = 1  # Top boundary (y = 1) -> c = 1
c[:, 0] = 0   # Bottom boundary (y = 0) -> c = 0

# Periodic boundary conditions in x-direction
def apply_periodic_bc(c):
    c[0, :] = c[-2, :]  # Left boundary wraps to second-last column
    c[-1, :] = c[1, :]  # Right boundary wraps to second column

# Gauss-Seidel Iteration
def solve_laplace_gauss_seidel(c, tol, max_iter):
    for iteration in range(max_iter):
        c_old = c.copy()  # Copy for convergence check

        # Update all inner points using Gauss-Seidel (in-place updates)
        for i in range(1, N-1):
            for j in range(1, N-1):
                c[i, j] = 0.25 * (c[i+1, j] + c[i-1, j] + c[i, j+1] + c[i, j-1])

        apply_periodic_bc(c)  # Apply periodic boundary conditions

        # Compute max difference to check for convergence
        diff = np.max(np.abs(c - c_old))
        if diff < tol:
            print(f"Converged in {iteration} iterations")
            break

    return c

# Solve the Laplace equation using Gauss-Seidel
c_steady = solve_laplace_gauss_seidel(c, tol, max_iter)

# Plot the final steady-state solution
plt.imshow(c_steady, origin="lower", extent=[0, 1, 0, 1], cmap="plasma")
plt.colorbar(label="Concentration")
plt.title("Steady-State Solution using Gauss-Seidel")
plt.xlabel("x")
plt.ylabel("y")
plt.show()