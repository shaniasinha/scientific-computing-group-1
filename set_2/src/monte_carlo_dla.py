import numpy as np
import matplotlib.pyplot as plt
import random

class RandomWalker:
    """
    Monte Carlo simulation of Diffusion-Limited Aggregation (DLA).
    Walkers start above the highest object and stick when contact occurs.
    """
    def __init__(self, grid_size=100, p_stick=1.0, max_object_size=500):
        self.height = grid_size
        self.width = grid_size
        self.p_stick = p_stick  
        self.object_size = 1  
        self.max_object_size = max_object_size  
        self.total_steps = 0  
        self.initialize()

    def initialize(self):
        """ Initializion of the grid, with the seed at the bottom """
        self.grid = np.zeros((self.width, self.height), dtype=int)
        self.grid[0, self.width // 2] = 1 
        self.place_walker()

    def place_walker(self):
        """Puts a walker five steps above the highest element of the cluster"""
        max_height = np.max(np.where(self.grid == 1)[1])  
        self.walker_y = min(max_height + 5, self.height - 1) 
        self.walker_x = random.randint(0, self.width - 1)  

        while self.grid[self.walker_y, self.walker_x] == 1:
            self.walker_x = random.randint(0, self.width - 1)
            self.walker_y = min(max_height + 5, self.height - 1)  

        self.grid[self.walker_y, self.walker_x] = 2

    def remove_walker(self):
        """ Deletes efectively the walker from the grid """
        if 0 <= self.walker_y < self.height and 0 <= self.walker_x < self.width:
            if self.grid[self.walker_y, self.walker_x] == 2:
                self.grid[self.walker_y, self.walker_x] = 0 

    def get_neighbours(self):
        """ Defines the neighbors of the random walker"""
        neighbours = []
        if self.walker_y > 0:
            neighbours.append((self.walker_x, self.walker_y - 1))#Down
        if self.walker_y < self.height - 1:
            neighbours.append((self.walker_x, self.walker_y + 1))#up
        if self.walker_x > 0:
            neighbours.append((self.walker_x - 1, self.walker_y))#left
        if self.walker_x < self.width - 1:
            neighbours.append((self.walker_x + 1, self.walker_y))#right
        return neighbours

    def next_step(self):
        
        """ Moves randomly the walker and sees if it should stick or not """
        added = False
        moved = False
        while not added and not moved :
            self.total_steps += 1  
            next_walker_x, next_walker_y = self.next_coordinates()
            moved = self.move(next_walker_x, next_walker_y)
          

            neighbours = self.get_neighbours()
            for neighbour in neighbours:
                if self.grid[neighbour[1], neighbour[0]] == 1 and np.random.random() <= self.p_stick:
                    self.grid[self.walker_y, self.walker_x] = 1  
                    self.object_size += 1  
                    added = True
                    self.place_walker()  
                    break

    def move(self, next_walker_x, next_walker_y):
        """moves randomly the walker. """
        if next_walker_y < 0 or next_walker_y >= self.height:
            self.remove_walker()
            self.place_walker()
            return True

        if next_walker_x < 0:
            next_walker_x = self.width - 1
        elif next_walker_x == self.width:
            next_walker_x = 0

        if self.grid[next_walker_y, next_walker_x] == 0:
            self.remove_walker()
            self.grid[next_walker_y, next_walker_x] = 2
            self.walker_x, self.walker_y = next_walker_x, next_walker_y
            return True
        return False

    def next_coordinates(self):
        """ Pick a random direction, they all carry the same weight. We also wrote a code that would give more weight to lateral directions """
        directions = ["left", "right", "up", "down"]
        weights = [0.25, 0.25, 0.25, 0.25] 
        direction = random.choices(directions, weights)[0]

        next_walker_x, next_walker_y = self.walker_x, self.walker_y

        if direction == "left":
            next_walker_x -= 1
        elif direction == "right":
            next_walker_x += 1
        elif direction == "up":
            next_walker_y += 1
        else:
            next_walker_y -= 1

        return next_walker_x, next_walker_y


def run_simulation():
    N = 100  
    max_object_size = 500  

    p_stick_values = [0.1, 0.4, 1.0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  
    for i, p_stick in enumerate(p_stick_values):
        dla = RandomWalker(N, p_stick, max_object_size)

        while dla.object_size < dla.max_object_size:
            dla.next_step()

        dla.remove_walker()

        axes[i].imshow(dla.grid, cmap='gray', origin='lower', interpolation='nearest')
        axes[i].set_title(f"Sticking probability $p_{{stick}} = {p_stick}$", fontsize=10)
        axes[i].set_xlabel("x-coordinate", fontsize=8)
        axes[i].set_ylabel("y-coordinate", fontsize=8)
        
    
    plt.show()
