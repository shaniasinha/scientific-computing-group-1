import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.special as sp

class TimeDependentDiffusion:
    def __init__(self, N, D=1.0, simulation_time=1, fig_name="diffusion", i_max=100, auto_adjust=True):
        if N < 2:
            raise ValueError("N must be at least 2.")
        if D <= 0:
            raise ValueError("D must be positive.")
        if simulation_time <= 0:
            raise ValueError("simulation_time must be positive.")
        
        self.D = D
        self.N = N
        self.dx = 1.0 / N
        self.dt = (self.dx**2) / (4 * self.D)
        self.dy = self.dx
        self.simulation_time = simulation_time
        self.fig_name = fig_name
        self.i_max = i_max 
        self.num_steps = int(simulation_time / self.dt)
        self.c = np.zeros((N, N), dtype=np.float64)
        self.c_simulation = np.zeros((self.num_steps, N, N), dtype=np.float64)

        self.is_stable()
        self.apply_boundary_conditions()

    def is_stable(self):
        """
        Check if the simulation is stable.

        >>> tdd = TimeDependentDiffusion(N=5, D=1.0)
        >>> tdd.is_stable()  # Should pass without error

        >>> tdd_unstable = TimeDependentDiffusion(N=5, D=10)
        Traceback (most recent call last):
            ...
        AssertionError: Stability condition not met.
        """
        stability = 4 * self.dt * self.D / self.dx**2
        if stability > 1:
            if self.auto_adjust:
                print(f"Adjusting dt for stability: {self.dt} -> {self.dt / stability}")
                self.dt /= stability
                self.num_steps = int(self.simulation_time / self.dt)
                self.c_simulation = np.zeros((self.num_steps, self.N, self.N), dtype=np.float64)  # Reinitialize
            else:
                raise AssertionError(f"Stability condition not met. dt={self.dt}, stability={stability}")

    def apply_boundary_conditions(self):
        """
        Apply boundary conditions.

        >>> tdd = TimeDependentDiffusion(N=3)
        >>> tdd.apply_boundary_conditions()
        >>> tdd.c[0, :]  # Bottom row should be 0
        array([0., 0., 0.])
        >>> tdd.c[-1, :]  # Top row should be 1
        array([1., 1., 1.])
        """
        self.c[-1, :] = 1  # Top boundary
        self.c[0, :] = 0   # Bottom boundary

    def run_time_stepping(self):
        """
        Run the time-stepping loop.

        >>> tdd = TimeDependentDiffusion(N=3, dt=0.001, D=1.0, simulation_time=0.01)
        >>> tdd.run_time_stepping()
        >>> np.all(tdd.c >= 0)  # Concentrations should not be negative
        True

        >>> sim = TimeDependentDiffusion(N=3, simulation_time=0.001)
        >>> sim.run_time_stepping()
        >>> sim.c_simulation.shape
        (1, 3, 3)
        """
        self.c_simulation[0] = np.copy(self.c)

        for t in range(1, self.num_steps):
            new_c = np.copy(self.c)

            # Update interior points using vectorized operations
            new_c[1:-1, 1:-1] = self.c[1:-1, 1:-1] + self.D * self.dt / self.dx**2 * (
                self.c[2:, 1:-1] + self.c[:-2, 1:-1] + self.c[1:-1, 2:] + self.c[1:-1, :-2] - 4 * self.c[1:-1, 1:-1]
            )

            # Handle periodic boundaries for left and right edges
            new_c[1:-1, 0] = self.c[1:-1, 0] + self.D * self.dt / self.dx**2 * (
                self.c[2:, 0] + self.c[:-2, 0] + self.c[1:-1, 1] + self.c[1:-1, -1] - 4 * self.c[1:-1, 0]
            )
            new_c[1:-1, -1] = self.c[1:-1, -1] + self.D * self.dt / self.dx**2 * (
                self.c[2:, -1] + self.c[:-2, -1] + self.c[1:-1, 0] + self.c[1:-1, -2] - 4 * self.c[1:-1, -1]
            )

            self.c = new_c
            self.apply_boundary_conditions()
            self.c_simulation[t] = np.copy(self.c)

            assert np.allclose(self.c[-1, :], 1), "Top row concentration is not 1"
            assert np.all(self.c >= 0), "Concentration is negative"


    def analytical_solution(self, x, t):
        """
        Compute analytical solution.

        >>> tdd = TimeDependentDiffusion(N=5)
        >>> x = np.array([0, 0.5, 1.0])
        >>> tdd.analytical_solution(x, 0)  # At t=0, should be zero everywhere
        array([0., 0., 0.])

        >>> tdd.analytical_solution(x, -1)  # Negative time, expect zeros or handle gracefully
        array([0., 0., 0.])
        """
        if t == 0:
            return np.zeros_like(x, dtype=np.float64)
        c = np.zeros_like(x, dtype=np.float64)
        for i in range(self.i_max + 1):
            term1 = sp.erfc((1 - x + 2 * i) / (2 * np.sqrt(self.D * t)))
            term2 = sp.erfc((1 + x + 2 * i) / (2 * np.sqrt(self.D * t)))
            c += term1 - term2
        return c
    
    def check_analytical_solution(self):
        """
        Compare numerical and analytical solutions for different time steps.
        Plots both solutions on the same figure for all time steps.

        Example:
        >>> tdd = TimeDependentDiffusion(N=50, dt=0.001, D=1.0, simulation_time=1.0)
        >>> tdd.run_time_stepping()
        >>> tdd.check_analytical_solution()
        """
        x = np.linspace(0, 1, self.N)
        times = [0.001, 0.01, 0.1, 1]

        plt.figure(figsize=(10, 6))

        for t in times:
            analytical = self.analytical_solution(x, t)
            timestep_index = int(t / self.dt)
            timestep_index = min(timestep_index, len(self.c_simulation) - 1)

            numerical = self.c_simulation[timestep_index, :, self.N // 2]

            plt.plot(x, analytical, linestyle="--", label=f"Analytical t={t:.3f}s")
            plt.plot(x, numerical, linestyle="-", label=f"Numerical t={t:.3f}s")

            error = np.abs(numerical - analytical)
            print(f"Maximum error at t = {t:.3f}s: {np.max(error):.5f}")

        plt.title("Comparison of Analytical and Numerical Solutions")
        plt.xlabel("x")
        plt.ylabel("Concentration (c)")
        plt.legend()
        plt.grid()
        plt.savefig(f"results/set_1/difussion/{self.fig_name}_analytical_vs_numerical.png")
        plt.show()

    def write_data(self, timestep):
        """
        Write the data to a CSV file
        """
        df = pd.DataFrame(self.c.flatten()).T
        df['timestep'] = timestep  
        if timestep == 0:
            df.to_csv(f"results/set_1/difussion/{self.fig_name}.csv", index=False, mode='w', header=True)
        else:
            df.to_csv(f"results/set_1/difussion/{self.fig_name}.csv", index=False, mode='a', header=False)

    def read_data(self):
        """
        Read the data from a CSV file
        """
        df = pd.read_csv(f"results/set_1/difussion/{self.fig_name}.csv")
        timesteps = df['timestep'].unique()
        self.c_simulation = np.zeros((len(timesteps), self.N, self.N))
        for timestep in timesteps:
            data = df[df['timestep'] == timestep].drop(columns=['timestep']).values.flatten()
            if len(data) == self.N * self.N:  
                self.c_simulation[int(timestep)] = data.reshape((self.N, self.N))
            else:
                print(f"Warning: Data size mismatch for timestep {timestep}. Expected {self.N * self.N}, got {len(data)}")

    def plot_results(self, timestep=0):
        """
        Plot the results of the diffusion simulation
        """
        actual_time = timestep * self.dt
        concentration = self.c_simulation[timestep]

        plt.figure(figsize=(8, 8))
        plt.imshow(concentration, extent=[0, 1, 0, 1], origin='lower', cmap='hot')
        plt.colorbar(label='Concentration')
        plt.title(f'Concentration at t = {actual_time:.2f}s') 
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f"results/set_1/difussion/{self.fig_name}_{timestep}.png")  
        plt.show()

    def create_animation(self, frame_skip=100):
        """
        Create an animation of the diffusion process.

        Parameters:
            frame_skip (int): Number of frames to skip between animation frames.
                            For example, frame_skip=100 means only every 100th frame is used.
        """
        if not hasattr(self, 'c_simulation') or len(self.c_simulation) == 0:
            raise ValueError("No simulation data found. Run the simulation first.")
        fig, ax = plt.subplots(figsize=(8, 8))
        img = ax.imshow(self.c_simulation[0], extent=[0, 1, 0, 1], origin='lower', cmap='hot', animated=True)
        plt.colorbar(img, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        def update(frame_index):
            actual_frame = frame_index * frame_skip
            frame_data = self.c_simulation[actual_frame]
            img.set_array(frame_data)
            ax.set_title(f'Time = {actual_frame * self.dt:.2f}s')
            return img,
    
        num_frames = len(self.c_simulation) // frame_skip
        ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)
        ani.save(f"results/set_1/difussion/difussion_animation{self.fig_name}.gif", writer="ffmpeg", fps=20)
        plt.close()

    def create_subplot_log_time(self):
        """
        Create 5x1 subplot, each showing the concentration at a different time step
        timestep = 0, 0.001, 0.01, 0.1, 1
        """
        fig, axs = plt.subplots(1, 5 , figsize=(20, 4))
        time_steps = [0, 0.001, 0.01, 0.1, 1]
        
        for i, ax in enumerate(axs):
            timestep_index = min(int(time_steps[i] / self.dt), len(self.c_simulation) - 1)
            if timestep_index < len(self.c_simulation):
                ax.imshow(self.c_simulation[timestep_index], extent=[0, 1, 0, 1], origin='lower', cmap='hot')
                ax.set_title(f'Time = {time_steps[i]:.3f}s')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
            else:
                ax.set_title(f'Time = {time_steps[i]:.3f}s (out of range)')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
        plt.tight_layout()
        plt.savefig(f"results/set_1/difussion/{self.fig_name}_subplots.png")
        plt.show()