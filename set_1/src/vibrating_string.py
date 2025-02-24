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

        >>> def psi(x): return np.sin(np.pi * x)
        >>> vs = VibratingString(psi, 100, dt=0.001, simulation_time=0.1)
        >>> vs.compute_second_time_step()
        >>> vs.u[1, 50]
        0.0004934802200549509
        """
        for i in range(1, self.N - 1):
            self.u[1, i] = self.u[0, i] + 0.5 * (self.c * self.dt / self.dx)**2 * (self.u[0, i + 1] - 2 * self.u[0, i] + self.u[0, i - 1])

    def run_time_stepping(self):
        """
        Run the time-stepping loop to compute the wave propagation over time

        >>> def psi(x): return np.sin(np.pi * x)
        >>> vs = VibratingString(psi, 100, dt=0.001, simulation_time=0.1)
        >>> vs.run_time_stepping()
        >>> vs.u[-1, 50]
        -0.0004934802200549509
        """
        for t in range(1, len(self.u) - 1):
            for i in range(1, self.N - 1):
                self.u[t + 1, i] = 2 * self.u[t, i] - self.u[t - 1, i] + (self.c * self.dt / self.dx)**2 * (self.u[t, i + 1] - 2 * self.u[t, i] + self.u[t, i - 1])

    def plot_static_simulation(self):
        """
        Plot the wave propagation over time using matplotlib static plot
        with a colormap for each time step and decreasing line thickness.
        """
        plt.figure(figsize=(8, 6))
        
        colormap = plt.cm.plasma
        num_steps = 10
        step_size = int(len(self.u) / num_steps)
        norm = plt.Normalize(vmin=0, vmax=num_steps)
        
        # Plot each time step with a color from the colormap and decreasing line thickness
        for i, t in enumerate(range(0, len(self.u), step_size)):
            color = colormap(norm(i))
            line_width = 5 * (1 - i / num_steps) + 2 
            plt.plot(self.x, self.u[t, :], color=color, alpha=1, linewidth=line_width, label=f"t = {t * self.dt:.3f}s")
        
        # Add the last step to the plot
        color = colormap(norm(num_steps))
        line_width = 1
        plt.plot(self.x, self.u[-1, :], color=color, alpha=1, linewidth=line_width, label=f"t = {self.simulation_time:.3f}s")

        plt.title("Wave Propagation Over Time")
        plt.xlabel("Position along the string")
        plt.ylabel("Displacement")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"results/set_1/wave/wave_static_{self.fig_name}.png", dpi=200)
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
        ani.save(f"results/set_1/wave/wave_animated_{self.fig_name}.gif", writer="ffmpeg", fps=10)
        plt.close(fig)
