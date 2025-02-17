
import numpy as np
import matplotlib.pyplot as plt

N = 50  
L = 1.0  
D = 1.0  
dx = L / N  
dt = (dx**2) / (4 * D)  
a = (dt * D) / (dx**2)  
steps = 10000  

c = np.zeros((N + 1, N + 1))

c[:, N] = 1  
c[:, 0] = 0  

for n in range(steps):
    c_new = c.copy()  

    for i in range(1, N):  
        for j in range(1, N):  
            c_new[i, j] = c[i, j] + a * (
                c[i+1, j] + c[i-1, j] + c[i, j+1] + c[i, j-1] - 4 * c[i, j]
            )
    
    for j in range(1, N):  
        c_new[0, j] = c[0, j] + a * (
            c[1, j] + c[N, j] + c[0, j+1] + c[0, j-1] - 4 * c[0, j]
        )
    
    for j in range(1, N):  
        c_new[N, j] = c[N, j] + a * (
            c[0, j] + c[N-1, j] + c[N, j+1] + c[N, j-1] - 4 * c[N, j]
        )
    
    c_new[:, 0] = 0  
    c_new[:, N] = 1  
    
    c = c_new.copy()

plt.imshow(c.T, cmap="hot", origin="lower", extent=[0, L, 0, L], vmin=0, vmax=1)
plt.colorbar(label="Concentration")
plt.title("Diffusion for one time unit")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
