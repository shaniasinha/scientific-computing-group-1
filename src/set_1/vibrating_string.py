import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class VibratingString:
    def __init__(self, psi_func, N, dt=0.001, simulation_time=0.1, fig_name="wave.png"):
        self.psi_func: function = psi_func
        self.fig_name = fig_name
        self.L = 1.0
        self.c = 1.0
        self.N = N
        self.dx = self.L / (self.N - 1)
        self.x = np.linspace(0, self.L, self.N)
        self.dt = dt
        self.simulation_time = simulation_time
        self.u = np.zeros((int(self.simulation_time / self.dt), self.N))

        self.grid = np.linspace(len(self.u), len(self.x))

        self.apply_boundary_conditions()
        self.fill_in_initial_conditions()
        self.compute_second_time_step()

    def apply_boundary_conditions(self):
        """
        Apply boundary conditions to the wave equation
        """
        self.u[:, 0] = 0
        self.u[:, -1] = 0

    def fill_in_initial_conditions(self):
        """"
        Fill in the initial conditions for the wave equation
        """
        self.u[0, :] = self.psi_func(self.x)

    def compute_second_time_step(self):
        """
        calculate the second time step using the Taylor series expansion
        """
        for i in range(1, self.N - 1):
            self.u[1, i] = self.u[0, i] + 0.5 * (self.c * self.dt / self.dx)**2 * (self.u[0, i + 1] - 2 * self.u[0, i] + self.u[0, i - 1])

    def run_time_stepping(self):
        """
        Run the time-stepping loop to compute the wave propagation over time
        """
        for t in range(1, len(self.u) - 1):
            for i in range(1, self.N - 1):
                self.u[t + 1, i] = 2 * self.u[t, i] - self.u[t - 1, i] + (self.c * self.dt / self.dx)**2 * (self.u[t, i + 1] - 2 * self.u[t, i] + self.u[t, i - 1])

    def plot_static_simulation(self):
        """
        Plot the wave propagation over time using matplotlib static plot
        """
        plt.figure(figsize=(10, 6))
        for t in range(0, len(self.u), int(len(self.u) / 10)): 
            plt.plot(self.x, self.u[t, :], label=f"t = {t * self.dt:.3f}s")

        plt.title("Wave Propagation Over Time")
        plt.xlabel("Position along the string")
        plt.ylabel("Displacement")
        plt.legend()
        plt.grid()
        plt.savefig(f"data/set_1/wave/wave_static_{self.fig_name}.png")
        plt.show()
        
    def plot_dynamic_simulation(self):
        """
        Plot the wave propagation over time using matplotlib animation
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot(self.x, self.u[0, :], label="Wave Motion")

        ax.set_title("Wave Propagation Over Time")
        ax.set_xlabel("Position along the string")
        ax.set_ylabel("Displacement")
        ax.legend()
        ax.grid()

        ax.set_ylim(np.min(self.u), np.max(self.u))
        def update(frame):
            line.set_ydata(self.u[frame, :])
            return line,

        ani = animation.FuncAnimation(fig, update, frames=range(0, len(self.u), 5), interval=50, blit=True)

        # Save animation as an MP4 or GIF file
        ani.save(f"data/set_1/wave/wave_animated_{self.fig_name}.gif", writer="ffmpeg", fps=10)
