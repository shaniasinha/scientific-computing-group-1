import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.special as sp

class TimeDependentDiffusion:
    def __init__(self, N, dt=0.001, D=1.0, simulation_time=1, fig_name="diffusion"):
        self.N = N
        self.dt = dt
        self.D = D
        self.dx = 1.0 / (N - 1)
        self.dy = 1.0 / (N - 1)
        self.simulation_time = simulation_time
        self.fig_name = fig_name
        self.num_steps = int(simulation_time / dt)
        self.c = np.zeros((N, N))
        self.c_simulation = np.zeros((self.num_steps, N, N))

        self.is_stable()
        self.apply_boundary_conditions()

    def is_stable(self):
        """
        Check if the simulation is stable
        """
        assert 4 * self.dt * self.D / self.dx**2 <= 1, "Stability condition not met."

    def apply_boundary_conditions(self):
        """
        Apply boundary conditions for the diffusion equation:
        y = N-1: c = 1 (top)
        y = 0: c = 0 (bottom)
        """
        self.c[-1, :] = 1  # Top boundary
        self.c[0, :] = 0   # Bottom boundary

    def run_time_stepping(self):
        """
        Run the time-stepping loop to compute the diffusion over time.
        This includes handling periodic boundary conditions for the left and right edges.
        """

        # Store the initial condition at t=0
        self.c_simulation[0] = np.copy(self.c)

        print(self.num_steps)
        for t in range(1, self.num_steps):  # Start from t=1 since t=0 is already stored
            new_c = np.copy(self.c)

            # Update interior points
            for i in range(1, self.N - 1):
                for j in range(1, self.N - 1):
                    new_c[i, j] = self.c[i, j] + self.D * self.dt / self.dx**2 * (
                        self.c[i+1, j] + self.c[i-1, j] + self.c[i, j+1] + self.c[i, j-1] - 4*self.c[i, j])

            # Handle periodic boundaries for left and right edges
            for i in range(1, self.N - 1):
                new_c[i, 0] = self.c[i, 0] + self.D * self.dt / self.dx**2 * (
                    self.c[i+1, 0] + self.c[i-1, 0] + self.c[i, 1] + self.c[i, self.N-1] - 4*self.c[i, 0])
                new_c[i, self.N-1] = self.c[i, self.N-1] + self.D * self.dt / self.dx**2 * (
                    self.c[i+1, self.N-1] + self.c[i-1, self.N-1] + self.c[i, 0] + self.c[i, self.N-2] - 4*self.c[i, self.N-1])

            self.c = new_c
            self.apply_boundary_conditions()

            # Store the concentration state at this timestep
            self.c_simulation[t] = np.copy(self.c)

            assert np.allclose(self.c[-1, :], 1), "Top row concentration is not 1"
            assert np.all(self.c >= 0), "Concentration is negative"


    def analytical_solution(self, x, t):
        c = np.zeros_like(x)
        for i in range(100):  # Sum over a reasonable number of terms
            term1 = sp.erfc((1 - x + 2 * i) / (2 * np.sqrt(self.D * t)))
            term2 = sp.erfc((1 + x + 2 * i) / (2 * np.sqrt(self.D * t)))
            c += term1 - term2
        return c
    
    def check_analytical_solution(self):
        x = np.linspace(0, 1, self.N)
        times = [0.001, 0.01, 0.1, 1]  # Example times to compare

        for t in times:
            analytical = self.analytical_solution(x, t)
            
            # Calculate the timestep index
            timestep_index = int(t / self.dt)
            
            # Ensure the timestep_index is within bounds
            if timestep_index >= len(self.c_simulation):
                timestep_index = len(self.c_simulation) - 1  # Use the last valid index
            
            numerical = self.c_simulation[timestep_index, :, self.N // 2]  # Middle row for simplicity

            plt.figure() 
            plt.plot(x, analytical, label='Analytical', color='orange')
            plt.scatter(x, numerical, label='Numerical')  
            plt.title(f'Comparison at t = {t:.3f}s')
            plt.xlabel('x')
            plt.ylabel('Concentration')
            plt.legend()
            plt.show()

            error = np.abs(numerical - analytical)
            print(f"Maximum error at t = {t:.3f}s: {np.max(error)}")

    def write_data(self, timestep):
        """
        Write the data to a CSV file
        """
        df = pd.DataFrame(self.c.flatten()).T
        df['timestep'] = timestep  
        if timestep == 0:
            df.to_csv(f"data/set_1/difussion/{self.fig_name}.csv", index=False, mode='w', header=True)
        else:
            df.to_csv(f"data/set_1/difussion/{self.fig_name}.csv", index=False, mode='a', header=False)

    def read_data(self):
        """
        Read the data from a CSV file
        """
        df = pd.read_csv(f"data/set_1/difussion/{self.fig_name}.csv")
        timesteps = df['timestep'].unique()
        self.c_simulation = np.zeros((len(timesteps), self.N, self.N))
        for timestep in timesteps:
            data = df[df['timestep'] == timestep].drop(columns=['timestep']).values.flatten()
            if len(data) == self.N * self.N:  # Ensure the data size is correct
                self.c_simulation[int(timestep)] = data.reshape((self.N, self.N))
            else:
                print(f"Warning: Data size mismatch for timestep {timestep}. Expected {self.N * self.N}, got {len(data)}")

    def plot_results(self, timestep=0):
        """
        Plot the results of the diffusion simulation
        """
        # Calculate the actual time from the timestep index
        actual_time = timestep * self.dt

        # Select the concentration data for the specified timestep
        concentration = self.c_simulation[timestep]
        print(concentration)

        plt.figure(figsize=(8, 8))
        plt.imshow(concentration, extent=[0, 1, 0, 1], origin='lower', cmap='hot')
        plt.colorbar(label='Concentration')
        plt.title(f'Concentration at t = {actual_time:.2f}s')  # Fixed title formatting
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f"data/set_1/difussion/{self.fig_name}_{timestep}.png")  # Save each plot with a unique name based on timestep
        plt.show()

    def create_animation(self, frame_skip=100):
        """
        Create an animation of the diffusion process.

        Parameters:
            frame_skip (int): Number of frames to skip between animation frames.
                            For example, frame_skip=100 means only every 100th frame is used.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        img = ax.imshow(self.c_simulation[0], extent=[0, 1, 0, 1], origin='lower', cmap='hot', animated=True)
        plt.colorbar(img, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        def update(frame_index):
            # Calculate the actual frame index in the simulation data
            actual_frame = frame_index * frame_skip
            frame_data = self.c_simulation[actual_frame]
            img.set_array(frame_data)
            ax.set_title(f'Time = {actual_frame * self.dt:.2f}s')
            return img,

        # Determine the number of frames to use in the animation
        num_frames = len(self.c_simulation) // frame_skip

        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)

        # Save animation as an MP4 or GIF file
        ani.save(f"data/set_1/difussion/difussion_animation{self.fig_name}.gif", writer="ffmpeg", fps=20)
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
            print(timestep_index)
            print(self.c_simulation.shape)
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
        plt.savefig(f"data/set_1/difussion/{self.fig_name}_subplots.png")
        plt.show()