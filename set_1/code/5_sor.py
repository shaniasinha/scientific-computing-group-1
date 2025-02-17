import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 50  # Grid size (N x N)
tol = 1e-5  # Convergence tolerance
max_iter = 10000  # Maximum number of iterations
omega = 1.85  # Optimal relaxation factor (typically between 1.7 and 1.9)

# Initialize concentration field
c = np.zeros((N, N))

# Apply boundary conditions
c[:, -1] = 1  # Top boundary (y = 1) -> c = 1
c[:, 0] = 0   # Bottom boundary (y = 0) -> c = 0

# Periodic boundary conditions in x-direction
def apply_periodic_bc(c):
    c[0, :] = c[-2, :]
    c[-1, :] = c[1, :]

# Successive Over-Relaxation (SOR)
def solve_laplace_sor(c, omega, tol, max_iter):
    for iteration in range(max_iter):
        c_old = c.copy()  # Copy for convergence check

        # Update all inner points using SOR
        for i in range(1, N-1):
            for j in range(1, N-1):
                c[i, j] = (1 - omega) * c[i, j] + (omega / 4) * (
                    c[i+1, j] + c[i-1, j] + c[i, j+1] + c[i, j-1]
                )

        apply_periodic_bc(c)  # Apply periodic boundary conditions

        # Compute max difference to check for convergence
        diff = np.max(np.abs(c - c_old))
        if diff < tol:
            print(f"Converged in {iteration} iterations with ω = {omega}")
            break

    return c

# Solve the Laplace equation using SOR
c_steady = solve_laplace_sor(c, omega, tol, max_iter)

# Plot the final steady-state solution
plt.imshow(c_steady, origin="lower", extent=[0, 1, 0, 1], cmap="plasma")
plt.colorbar(label="Concentration")
plt.title(f"Steady-State Solution using SOR (ω = {omega})")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
