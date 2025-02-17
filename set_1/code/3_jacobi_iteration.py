import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 50  # Grid size (N x N)
tol = 1e-5  # Convergence tolerance
max_iter = 10000  # Maximum number of iterations

# Initialize concentration field
c = np.zeros((N, N))

# Apply boundary conditions
c[:, -1] = 1  # c(x, y=1) = 1 (top boundary)
c[:, 0] = 0   # c(x, y=0) = 0 (bottom boundary)

# Periodic boundary conditions in x-direction
def apply_periodic_bc(c):
    c[0, :] = c[-2, :]
    c[-1, :] = c[1, :]

# Jacobi Iteration Method
def solve_laplace_jacobi(c, tol, max_iter):
    c_new = np.copy(c)
    for iteration in range(max_iter):
        c_old = c_new.copy()

        # Update all inner points
        for i in range(1, N-1):
            for j in range(1, N-1):
                c_new[i, j] = 0.25 * (c_old[i+1, j] + c_old[i-1, j] + c_old[i, j+1] + c_old[i, j-1])

        apply_periodic_bc(c_new)

        # Compute max difference to check for convergence
        diff = np.max(np.abs(c_new - c_old))
        if diff < tol:
            print(f"Converged in {iteration} iterations")
            break

    return c_new

# Solve the Laplace equation
c_steady = solve_laplace_jacobi(c, tol, max_iter)

# Plot the final steady-state solution
plt.imshow(c_steady, origin="lower", extent=[0, 1, 0, 1], cmap="plasma")
plt.colorbar(label="Concentration")
plt.title("Steady-State Solution of Diffusion (Laplace Equation)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
