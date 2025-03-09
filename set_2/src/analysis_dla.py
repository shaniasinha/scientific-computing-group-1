import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import the DLA class and numba_sor function from your existing code
# Assuming dla.py contains your DLA implementation
from src.dla import DLASimulation, numba_sor

def analyze_omega_performance(N=100, num_particles=20, omega_values=None):
    """
    Analyze the performance of different omega values for the SOR method in DLA simulation.
    """
    if omega_values is None:
        # Create a range of omega values to test (focusing on the region of interest)
        omega_values = np.linspace(1.6, 1.99, 5)
    
    total_iterations = np.zeros_like(omega_values)
    execution_times = np.zeros_like(omega_values)
    
    for i, omega in enumerate(omega_values):
        print(f"Testing omega = {omega:.2f}")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create DLA simulation with current omega
        dla = DLASimulation(N=N, max_iter_sor=5000, tol=1e-5, omega=omega)
        
        # Track total SOR iterations
        total_iters = 0
        
        # Run simulation for a fixed number of particles
        for _ in tqdm(range(num_particles-1)):  # -1 because we already have the seed
            # Solve Laplace equation with current boundaries
            dla.apply_boundary_conditions()
            iters = dla.solve_laplace_sor()
            total_iters += iters
            
            # Add a new particle
            success = dla.grow_cluster()
            if not success:
                print("Unable to add more particles. Simulation stopped.")
                break
        
        # Store the total number of iterations
        total_iterations[i] = total_iters
        
    return omega_values, total_iterations, execution_times

def find_optimal_omega_vs_gridsize(N_values=None, num_particles=1000):
    """
    Find the optimal omega value for different grid sizes.
    """
    if N_values is None:
        # Create a range of grid sizes to test
        N_values = np.array([20, 40, 60, 80, 100, 120, 140])
    
    # Initialize arrays for results
    optimal_omegas = np.zeros_like(N_values, dtype=float)
    
    # Calculate theoretical optimal omega values
    theoretical_omegas = 2.0 / (1.0 + np.sin(np.pi / N_values))
    
    for i, N in enumerate(N_values):
        print(f"Testing grid size N = {N}")
        
        # For each grid size, test a range of omega values centered around the theoretical optimum
        theo_omega = theoretical_omegas[i]
        omega_range = np.linspace(1.5, min(1.99, theo_omega + 0.1), 15)
        
        # Test these omega values
        omega_vals, total_iters, _ = analyze_omega_performance(N=N, num_particles=num_particles, 
                                                             omega_values=omega_range)
        
        # Find the omega that minimizes total iterations
        min_iter_idx = np.argmin(total_iters)
        optimal_omegas[i] = omega_vals[min_iter_idx]
        
        print(f"Grid size N = {N}, Optimal omega = {optimal_omegas[i]:.4f}, "
              f"Theoretical omega = {theoretical_omegas[i]:.4f}")
    
    return N_values, optimal_omegas, theoretical_omegas

def plot_results(omega_values, total_iterations, execution_times, N):
    """
    Plot the results of the omega performance analysis.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot iterations vs omega
    plt.subplot(2, 1, 1)
    plt.plot(omega_values, total_iterations, 'bo-', linewidth=2)
    plt.xlabel('Omega (ω)', fontsize=12)
    plt.ylabel('Total SOR Iterations', fontsize=12)
    plt.title(f'Total SOR Iterations vs Omega (N={N})', fontsize=14)
    plt.grid(True)
    
    # Highlight the minimum
    min_iter_idx = np.argmin(total_iterations)
    min_omega = omega_values[min_iter_idx]
    min_iters = total_iterations[min_iter_idx]
    plt.plot(min_omega, min_iters, 'ro', markersize=10)
    plt.annotate(f'Min at ω = {min_omega:.3f}\n({min_iters:.0f} iterations)', 
                 xy=(min_omega, min_iters),
                 xytext=(min_omega + 0.1, min_iters * 1.1),
                 arrowprops=dict(facecolor='red', shrink=0.05))
    
    # Plot execution time vs omega
    plt.subplot(2, 1, 2)
    plt.plot(omega_values, execution_times, 'go-', linewidth=2)
    plt.xlabel('Omega (ω)', fontsize=12)
    plt.ylabel('Execution Time (s)', fontsize=12)
    plt.title(f'Execution Time vs Omega (N={N})', fontsize=14)
    plt.grid(True)
    
    # Highlight the minimum
    min_time_idx = np.argmin(execution_times)
    min_time_omega = omega_values[min_time_idx]
    min_time = execution_times[min_time_idx]
    plt.plot(min_time_omega, min_time, 'ro', markersize=10)
    plt.annotate(f'Min at ω = {min_time_omega:.3f}\n({min_time:.2f} s)', 
                 xy=(min_time_omega, min_time),
                 xytext=(min_time_omega + 0.1, min_time * 1.1),
                 arrowprops=dict(facecolor='red', shrink=0.05))
    
    plt.tight_layout()
    plt.savefig(f'omega_performance_N{N}.png', dpi=150)
    plt.show()

def plot_optimal_omega_vs_N(N_values, optimal_omegas, theoretical_omegas):
    """
    Plot the optimal omega values versus grid size.
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(N_values, optimal_omegas, 'bo-', linewidth=2, label='Empirical Optimal ω')
    plt.plot(N_values, theoretical_omegas, 'r--', linewidth=2, label='Theoretical ω = 2/(1+sin(π/N))')
    
    plt.xlabel('Grid Size (N)', fontsize=12)
    plt.ylabel('Optimal Omega (ω)', fontsize=12)
    plt.title('Optimal SOR Relaxation Parameter vs Grid Size', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    
    # Add text showing formula
    plt.text(0.5, 0.05, 'Theoretical optimal ω = 2/(1+sin(π/N))', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('optimal_omega_vs_gridsize.png', dpi=150)
    plt.show()