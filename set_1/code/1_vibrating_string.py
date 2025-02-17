# Implement discretized 1-D wave equation for a vibrating string

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set parameters
L = 1.0             # Length of the string
c = 1.0             # Wave velocity
T = 2.0             # Total time
N = 100             # Number of spatial points

dx = L/N            # Spatial step size
dt = 0.001          # Time step size

M = int(T/dt)       # Number of time steps

# Initialize arrays
u = np.zeros((N, M))        # u(x, t)
x = np.linspace(0, L, N)    # x values

# Set initial conditions
# Problem A.i) u(x, 0) = sin(2*pi*x)
u[:, 0] = np.sin(2*np.pi*x)
# Problem A.ii) u(x, 0) = sin(5*pi*x)
# u[:, 0] = np.sin(5*np.pi*x)
# Problem A.iii) u(x, 0) = np.sin(5*np.pi*x) if 0.2 < x < 0.4, 0 otherwise
# u[int(0.2*N):int(0.4*N), 0] = np.sin(5*np.pi*x[int(0.2*N):int(0.4*N)])

# Set boundary conditions
# u(x=0, t) = u(x=L, t) = 0
u[0, :] = 0
u[-1, :] = 0

# Implement central difference in time and space
for n in range(0, M-1):
    for i in range(1, N-1):
        u[i, n+1] = 2*u[i, n] - u[i, n-1] + (c*dt/dx)**2 * (u[i+1, n] - 2*u[i, n] + u[i-1, n])

# Plot the solution
# fig, ax = plt.subplots()
# ax.set_xlim(0, L)
# ax.set_ylim(-1.2, 1.2)
# ax.set_xlabel("x")
# ax.set_ylabel("u(x,t)")
# ax.set_title("Vibrating String")

# line, = ax.plot([], [], 'b-', lw=2)
# frames = int(T / dt)

# def init():
#     line.set_data([], [])
#     return line,

# def update(frame):
#     line.set_data(x, u[:, frame])
#     return line,

# ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=30)

# plt.show()