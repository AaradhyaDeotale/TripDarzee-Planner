import math
import pandas as pd
import random
import time as tm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pso_tsp import SolveTSPUsingPSO

csv_filename = 'pso_out.csv'
data = pd.read_csv(csv_filename)
# =============================================================================
# Extend the PSO class to record convergence (best distance per iteration)
# =============================================================================
class SolveTSPUsingPSOConvergence(SolveTSPUsingPSO):
    def run(self):
        convergence_curve = []
        start_time = tm.time()
        for _ in range(self.iterations):
            for particle in self.particles:
                particle.clear_velocity()
                temp_velocity = []
                pbest_route = particle.pbest
                gbest_route = self.gbest

                new_route = particle.route.copy()

                # Generate swaps to move toward the particle's best (pbest)
                for i in range(len(new_route)):
                    if new_route[i] != pbest_route[i]:
                        j = pbest_route.index(new_route[i])
                        swap = (i, j, self.pbest_prob)
                        temp_velocity.append(swap)
                        new_route[i], new_route[j] = new_route[j], new_route[i]

                # Generate swaps to move toward the global best (gbest)
                for i in range(len(new_route)):
                    if new_route[i] != gbest_route[i]:
                        j = gbest_route.index(new_route[i])
                        swap = (i, j, self.gbest_prob)
                        temp_velocity.append(swap)
                        new_route[i], new_route[j] = new_route[j], new_route[i]

                particle.velocity = temp_velocity

                # Apply the swaps to the original route with their given probabilities
                new_route = particle.route.copy()
                for swap in temp_velocity:
                    if random.random() <= swap[2]:
                        new_route[swap[0]], new_route[swap[1]] = new_route[swap[1]], new_route[swap[0]]
                particle.route = new_route

                # Calculate the new tour distance
                particle.current_cost = self._calculate_distance(new_route)
                particle.update_pbest()
                # Update global best if this particle found a better route
                if particle.pbest_cost < self.gbest_distance:
                    self.gbest = particle.pbest.copy()
                    self.gbest_distance = particle.pbest_cost
            convergence_curve.append(self.gbest_distance)
        runtime = tm.time() - start_time
        return runtime, self.gbest_distance, convergence_curve

# =============================================================================
# Experiment functions
# =============================================================================
def run_experiments(num_runs=30, population_size=15, iterations=100, num_nodes=30):
    # Create a fixed TSP instance: generate 'num_nodes' random nodes
    nodes = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_nodes)]
    
    best_distances = []
    runtimes = []
    convergence_data = []
    
    for _ in tqdm(range(num_runs), desc='Experiment Runs'):
        pso = SolveTSPUsingPSOConvergence(population_size=population_size,
                                           iterations=iterations,
                                           pbest_prob=0.9,
                                           gbest_prob=0.1,
                                           nodes=nodes)
        runtime, best_distance, conv_curve = pso.run()
        best_distances.append(best_distance)
        runtimes.append(runtime)
        convergence_data.append(conv_curve)
    return nodes, best_distances, runtimes, convergence_data

def plot_results(best_distances, runtimes, convergence_data, iterations):
    # -------------------------------
    # Histogram: Best Tour Distances
    # -------------------------------
    plt.figure(figsize=(10, 6))
    plt.hist(best_distances, bins=10, color='skyblue', edgecolor='black')
    plt.title('Histogram of Best Tour Distances')
    plt.xlabel('Tour Distance')
    plt.ylabel('Frequency')
    plt.show()

    # -------------------------------
    # Histogram: Runtimes
    # -------------------------------
    plt.figure(figsize=(10, 6))
    plt.hist(runtimes, bins=10, color='lightgreen', edgecolor='black')
    plt.title('Histogram of Runtimes')
    plt.xlabel('Runtime (seconds)')
    plt.ylabel('Frequency')
    plt.show()

    # -------------------------------
    # Convergence Curves: All Runs & Average
    # -------------------------------
    plt.figure(figsize=(10, 6))
    for curve in convergence_data:
        plt.plot(range(1, iterations+1), curve, alpha=0.3, color='gray')
    avg_curve = np.mean(convergence_data, axis=0)
    plt.plot(range(1, iterations+1), avg_curve, color='red', linewidth=2, label='Average Convergence')
    plt.title('Convergence Curves')
    plt.xlabel('Iteration')
    plt.ylabel('Best Tour Distance')
    plt.legend()
    plt.show()

def print_statistics(best_distances, runtimes):
    best_distances = np.array(best_distances)
    runtimes = np.array(runtimes)
    
    print("=== Statistics over Runs ===")
    print(f"Mean Best Distance: {np.mean(best_distances):.2f}")
    print(f"Std Dev of Best Distance: {np.std(best_distances):.2f}")
    print(f"Min Best Distance: {np.min(best_distances):.2f}")
    print(f"Max Best Distance: {np.max(best_distances):.2f}")
    print("")
    print(f"Mean Runtime: {np.mean(runtimes):.2f} s")
    print(f"Std Dev of Runtime: {np.std(runtimes):.2f} s")
    print(f"Min Runtime: {np.min(runtimes):.2f} s")
    print(f"Max Runtime: {np.max(runtimes):.2f} s")

# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    # Experiment parameters
    NUM_RUNS = 30          # number of independent runs
    POPULATION_SIZE = 15   # number of particles in the swarm
    ITERATIONS = 100       # iterations per run
    NUM_NODES = 30         # number of cities/nodes in the TSP instance

    # Run experiments
    nodes, best_distances, runtimes, convergence_data = run_experiments(
        num_runs=NUM_RUNS,
        population_size=POPULATION_SIZE,
        iterations=ITERATIONS,
        num_nodes=NUM_NODES
    )

    # Print out tangible statistics
    print_statistics(best_distances, runtimes)

    # Plot results
    plot_results(best_distances, runtimes, convergence_data, ITERATIONS)
