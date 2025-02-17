from scipy.special import erfc
import numpy as np
import matplotlib.pyplot as plt


N = 50  
L = 1.0  
D = 1.0  
dx = L / N 
dt = (dx**2) / (4 * D)  
a = (dt * D) / (dx**2)  
steps_per_time = int(1.0 / dt) 

times = [0.001, 0.01, 0.1, 1.0]  

#A choice was made to truncate the series to five terms only
def analytic_solution(y, t, D, i_max=5):
    c = np.zeros_like(y)
    for i in range(i_max + 1): 
        c += erfc((1 - y + 2 * i) / (2 * np.sqrt(D * t))) - erfc((1 + y + 2 * i) / (2 * np.sqrt(D * t)))
    return c

c = np.zeros((N + 1, N + 1))
c[:, N] = 1  
c[:, 0] = 0  

y = np.linspace(0, 1, N + 1)

plt.figure(figsize=(10, 6))

for t in times:
 
    steps = int(t / dt)
    
   
    for _ in range(steps):
        c_new = c.copy()
        
       
        for i in range(1, N):
            for j in range(1, N):
                c_new[i, j] = c[i, j] + a * (
                    c[i+1, j] + c[i-1, j] + c[i, j+1] + c[i, j-1] - 4 * c[i, j])
        
        
        for j in range(1, N):
            c_new[0, j] = c[0, j] + a * (
                c[1, j] + c[N, j] + c[0, j+1] + c[0, j-1] - 4 * c[0, j])
            c_new[N, j] = c[N, j] + a * (
                c[0, j] + c[N-1, j] + c[N, j+1] + c[N, j-1] - 4 * c[N, j])
        
       
        c_new[:, 0] = 0
        c_new[:, N] = 1
        
        c = c_new.copy()
    
    c_analytic = analytic_solution(y, t, D)  
    c_numeric = c[int(N / 2), :] #We write this because the diffusion only depends of y, so x can be chosen arbitrarily, and we just decide which x we want to observe the simulation at 
    
    plt.plot(y, c_analytic, label=f"Analytic {t}", linestyle="--")
    plt.plot(y, c_numeric, label=f"Time_Dependent {t}", linestyle="-.")

plt.title("Comparison of Analytic and Numeric Solutions")
plt.xlabel("y")
plt.ylabel("Concentration (c)")
plt.legend()
plt.grid()
plt.show()
