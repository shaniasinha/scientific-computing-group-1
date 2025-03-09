import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

class GrayScottModel:
    def __init__(self, N=100, Du=0.16, Dv=0.08, f=0.035, k=0.060):
        """
        Initialize the Gray–Scott model with the given parameters.
        """
        self.N = N
        self.Du = Du
        self.Dv = Dv
        self.f = f
        self.k = k

        self.dx = 1.0
        self.dt = 0.2 * self.dx**2 / max(self.Du, self.Dv)

        self.u = np.ones((N, N))
        self.v = np.zeros((N, N))

        center = N // 2
        radius = N // 10
        self.u[center - radius : center + radius,
               center - radius : center + radius] = 0.5
        self.v[center - radius : center + radius,
               center - radius : center + radius] = 0.25

        self.u += 0.01 * np.random.random((N, N))
        self.v += 0.01 * np.random.random((N, N))

        self.iterations = 0
        self.frames = []
        self.simulation = []

    def laplacian(self, field):
        """Compute the Laplacian using finite differences with Neumann BCs."""
        lap = np.zeros_like(field)

        lap[1:-1, 1:-1] = (
            field[0:-2, 1:-1] + field[2:, 1:-1] +
            field[1:-1, 0:-2] + field[1:-1, 2:] -
            4 * field[1:-1, 1:-1]
        ) / (self.dx**2)

        lap[0, 1:-1] = (
            field[1, 1:-1] +
            field[0, 0:-2] + field[0, 2:] -
            3 * field[0, 1:-1]
        ) / (self.dx**2)

        lap[-1, 1:-1] = (
            field[-2, 1:-1] +
            field[-1, 0:-2] + field[-1, 2:] -
            3 * field[-1, 1:-1]
        ) / (self.dx**2)

        lap[1:-1, 0] = (
            field[0:-2, 0] + field[2:, 0] +
            field[1:-1, 1] -
            3 * field[1:-1, 0]
        ) / (self.dx**2)

        lap[1:-1, -1] = (
            field[0:-2, -1] + field[2:, -1] +
            field[1:-1, -2] -
            3 * field[1:-1, -1]
        ) / (self.dx**2)

        lap[0, 0] = (field[1, 0] + field[0, 1] - 2 * field[0, 0]) / (self.dx**2)
        lap[0, -1] = (field[1, -1] + field[0, -2] - 2 * field[0, -1]) / (self.dx**2)
        lap[-1, 0] = (field[-2, 0] + field[-1, 1] - 2 * field[-1, 0]) / (self.dx**2)
        lap[-1, -1] = (field[-2, -1] + field[-1, -2] - 2 * field[-1, -1]) / (self.dx**2)

        return lap

    def step(self):
        """Perform a single time integration step using Forward Euler."""
        lap_u = self.laplacian(self.u)
        lap_v = self.laplacian(self.v)

        uvv = self.u * self.v**2

        u_new = self.u + self.dt * (self.Du * lap_u - uvv + self.f * (1 - self.u))
        v_new = self.v + self.dt * (self.Dv * lap_v + uvv - (self.f + self.k) * self.v)

        self.u = np.clip(u_new, 0, 1)
        self.v = np.clip(v_new, 0, 1)

        self.iterations += 1

    def run(self, steps=1000, save_interval=10):
        """
        Run the simulation for a specified number of steps.
        Every save_interval steps, a snapshot of u and v is stored.
        The snapshots are saved to the 'simulation' attribute.
        """
        self.simulation = []
        for i in range(steps):
            self.step()

            if i % save_interval == 0 or i == steps - 1:
                self.simulation.append((self.u.copy(), self.v.copy()))
        self.frames = self.simulation.copy()

    def plot(self):
        """Plot the current state of u and v."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        im1 = ax1.imshow(self.u, cmap="viridis", vmin=0, vmax=1)
        ax1.set_title(f"Chemical U (t={self.iterations})")
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(self.v, cmap="magma", vmin=0, vmax=0.5)
        ax2.set_title(f"Chemical V (t={self.iterations})")
        plt.colorbar(im2, ax=ax2)

        plt.tight_layout()
        plt.show()

    def create_animation(self, frames=None, interval=50):
        """
        Create an animation of the system evolution.
        If frames is None, animates all stored snapshots.
        """
        if frames is None:
            frames = len(self.frames)
        frames = min(frames, len(self.frames))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        im1 = ax1.imshow(
            self.frames[0][0],
            cmap="viridis",
            vmin=0,
            vmax=1,
            animated=True,
        )
        ax1.set_title("Chemical U")
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(
            self.frames[0][1],
            cmap="magma",
            vmin=0,
            vmax=0.5,
            animated=True,
        )
        ax2.set_title("Chemical V")
        plt.colorbar(im2, ax=ax2)

        plt.tight_layout()

        def update(frame):
            u_frame, v_frame = self.frames[frame]
            im1.set_array(u_frame)
            im2.set_array(v_frame)
            ax1.set_title(
                f"Chemical U (t={frame*(self.iterations // max(len(self.frames),1))})"
            )
            ax2.set_title(
                f"Chemical V (t={frame*(self.iterations // max(len(self.frames),1))})"
            )
            return im1, im2

        ani = FuncAnimation(
            fig, update, frames=frames, interval=interval, blit=True
        )
        plt.close()
        return ani
