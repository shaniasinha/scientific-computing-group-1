import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
N = 50  # Grid size (N x N)
D = 1.0  # Diffusion coefficient
dx = 1.0 / N  # Grid spacing
dt = 0.25 * dx**2 / D  # Time step (ensuring stability with 4Dt/dx² <= 1)
T_final = 1  # Final time
num_steps = int(T_final / dt)  # Number of time steps

# Initialize concentration field
c = np.zeros((N, N))

# Apply boundary conditions
c[:, -1] = 1  # c(x, y=1; t) = 1
c[:, 0] = 0   # c(x, y=0; t) = 0

# Periodic boundary conditions in the x-direction
def apply_periodic_bc(c):
    c[0, :] = c[-2, :]  # c(x=0) = c(x=N-1)
    c[-1, :] = c[1, :]  # c(x=N) = c(x=1)

# Update function using the explicit finite difference scheme
def diffusion_step(c):
    c_new = np.copy(c)
    for i in range(1, N-1):
        for j in range(1, N-1):
            c_new[i, j] = c[i, j] + (D * dt / dx**2) * (
                c[i+1, j] + c[i-1, j] + c[i, j+1] + c[i, j-1] - 4 * c[i, j]
            )
    apply_periodic_bc(c_new)
    return c_new

# Run simulation and store snapshots for plotting
snapshots = []
for step in range(num_steps):
    c = diffusion_step(c)
    if step in [0, int(0.001/dt), int(0.01/dt), int(0.1/dt), int(1/dt)]:
        snapshots.append(np.copy(c))

# Plot snapshots at different times
fig, axes = plt.subplots(1, len(snapshots), figsize=(15, 5))
times = [0, 0.001, 0.01, 0.1, 1]

for ax, snap, time in zip(axes, snapshots, times):
    im = ax.imshow(snap, origin="lower", extent=[0, 1, 0, 1], cmap="plasma")
    ax.set_title(f"t = {time:.3f}")
    fig.colorbar(im, ax=ax)

plt.show()

# Animation
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(c, origin="lower", extent=[0, 1, 0, 1], cmap="plasma")

def update(frame):
    global c
    c = diffusion_step(c)
    im.set_array(c)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=200, interval=50)
plt.show()
