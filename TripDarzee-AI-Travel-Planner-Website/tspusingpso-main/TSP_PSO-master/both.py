import math
import random
import time as tm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# PSO-TSP Implementation
# =============================================================================
class SolveTSPUsingPSO:
    class Particle:
        def __init__(self, route, cost):
            self.route = route
            self.pbest = route.copy()
            self.current_cost = cost
            self.pbest_cost = cost
            self.velocity = []

        def clear_velocity(self):
            self.velocity = []

        def update_pbest(self):
            if self.current_cost < self.pbest_cost:
                self.pbest = self.route.copy()
                self.pbest_cost = self.current_cost

    def __init__(self, population_size=10, iterations=100, pbest_prob=0.9, gbest_prob=0.1, nodes=None, labels=None):
        self.population_size = population_size
        self.iterations = iterations
        self.pbest_prob = pbest_prob
        self.gbest_prob = gbest_prob
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.labels = labels if labels is not None else list(range(1, self.num_nodes + 1))
        self.particles = []
        self.gbest = None
        self.gbest_distance = float('inf')
        self._initialize_particles()

    def _initialize_particles(self):
        # Generate random particles (population_size - 1)
        for _ in range(self.population_size - 1):
            route = random.sample(range(self.num_nodes), self.num_nodes)
            distance = self._calculate_distance(route)
            self.particles.append(self.Particle(route, distance))
        # Generate a greedy route particle
        greedy_route = self._greedy_route()
        greedy_distance = self._calculate_distance(greedy_route)
        self.particles.append(self.Particle(greedy_route, greedy_distance))
        # Update global best
        for particle in self.particles:
            if particle.pbest_cost < self.gbest_distance:
                self.gbest = particle.pbest.copy()
                self.gbest_distance = particle.pbest_cost

    def _greedy_route(self, start_index=0):
        unvisited = list(range(self.num_nodes))
        current = start_index
        unvisited.remove(current)
        route = [current]
        while unvisited:
            next_node = min(unvisited, key=lambda x: self._distance(current, x))
            route.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        return route

    def _distance(self, a, b):
        dx = self.nodes[a][0] - self.nodes[b][0]
        dy = self.nodes[a][1] - self.nodes[b][1]
        return math.sqrt(dx ** 2 + dy ** 2)

    def _calculate_distance(self, route):
        total = 0.0
        for i in range(len(route)):
            total += self._distance(route[i], route[(i + 1) % len(route)])
        return total

    def run(self):
        # Basic run without convergence tracking
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
                new_route = particle.route.copy()
                for swap in temp_velocity:
                    if random.random() <= swap[2]:
                        new_route[swap[0]], new_route[swap[1]] = new_route[swap[1]], new_route[swap[0]]
                particle.route = new_route
                particle.current_cost = self._calculate_distance(new_route)
                particle.update_pbest()
                if particle.pbest_cost < self.gbest_distance:
                    self.gbest = particle.pbest.copy()
                    self.gbest_distance = particle.pbest_cost
        runtime = tm.time() - start_time
        return runtime, self.gbest_distance

# Extend PSO for convergence tracking (records best distance at each iteration)
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
                # Generate swaps for pbest
                for i in range(len(new_route)):
                    if new_route[i] != pbest_route[i]:
                        j = pbest_route.index(new_route[i])
                        swap = (i, j, self.pbest_prob)
                        temp_velocity.append(swap)
                        new_route[i], new_route[j] = new_route[j], new_route[i]
                # Generate swaps for gbest
                for i in range(len(new_route)):
                    if new_route[i] != gbest_route[i]:
                        j = gbest_route.index(new_route[i])
                        swap = (i, j, self.gbest_prob)
                        temp_velocity.append(swap)
                        new_route[i], new_route[j] = new_route[j], new_route[i]
                particle.velocity = temp_velocity
                new_route = particle.route.copy()
                for swap in temp_velocity:
                    if random.random() <= swap[2]:
                        new_route[swap[0]], new_route[swap[1]] = new_route[swap[1]], new_route[swap[0]]
                particle.route = new_route
                particle.current_cost = self._calculate_distance(new_route)
                particle.update_pbest()
                if particle.pbest_cost < self.gbest_distance:
                    self.gbest = particle.pbest.copy()
                    self.gbest_distance = particle.pbest_cost
            convergence_curve.append(self.gbest_distance)
        runtime = tm.time() - start_time
        return runtime, self.gbest_distance, convergence_curve

# =============================================================================
# Experiment & Analysis Functions
# =============================================================================
def run_experiments(num_runs=30, population_size=20, iterations=100, num_nodes=30, output_csv='pso_out.csv'):
    """
    Run multiple independent runs on a fixed TSP instance.
    For each run, record per-iteration best distance, total runtime, etc.
    Save all data to output_csv.
    """
    results = []
    # Create a fixed TSP instance (same nodes for all runs)
    nodes = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_nodes)]
    
    for run in tqdm(range(num_runs), desc='Experiment Runs'):
        pso = SolveTSPUsingPSOConvergence(population_size=population_size,
                                          iterations=iterations,
                                          pbest_prob=0.9,
                                          gbest_prob=0.1,
                                          nodes=nodes)
        run_runtime, best_distance, convergence_curve = pso.run()
        # Record each iteration's best distance
        for iteration, distance in enumerate(convergence_curve, start=1):
            results.append({
                'Iteration': iteration,
                'Population': population_size,
                'Time': run_runtime,  # total runtime for the run
                'Distance': distance
            })
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    return df, nodes

def analyze_results(df, optimal_value=None):
    """
    Analyze the CSV data and produce tangible measures:
    - Overall statistics
    - Histograms of tour distances and runtimes
    - Convergence curve (mean and std per iteration)
    - Optional optimality gap analysis
    """
    # Print overall statistics
    print("=== Overall Statistics ===")
    print(df.describe())
    
    # Histogram: Best Tour Distances
    plt.figure(figsize=(10, 6))
    plt.hist(df['Distance'], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Tour Distance')
    plt.ylabel('Frequency')
    plt.title('Histogram of PSO Tour Distances')
    plt.show()
    
    # Histogram: Runtimes
    plt.figure(figsize=(10, 6))
    plt.hist(df['Time'], bins=20, color='salmon', edgecolor='black')
    plt.xlabel('Runtime (seconds)')
    plt.ylabel('Frequency')
    plt.title('Histogram of PSO Runtimes')
    plt.show()
    
    # Convergence Plot: Average Tour Distance vs. Iteration
    convergence = df.groupby('Iteration').agg({
        'Distance': ['mean', 'std', 'min', 'max']
    })
    convergence.columns = ['mean_distance', 'std_distance', 'min_distance', 'max_distance']
    convergence = convergence.reset_index()
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(convergence['Iteration'], 
                 convergence['mean_distance'], 
                 yerr=convergence['std_distance'], 
                 fmt='-o', capsize=5, label='Mean Distance ± Std Dev')
    plt.xlabel('Iteration')
    plt.ylabel('Tour Distance')
    plt.title('Convergence Curve: Average Tour Distance vs. Iteration')
    plt.legend()
    plt.show()
    
    # Additional Analysis: Optimality Gap, if optimal_value is provided
    if optimal_value is not None:
        df['Optimality Gap (%)'] = (df['Distance'] - optimal_value) / optimal_value * 100
        plt.figure(figsize=(10, 6))
        plt.hist(df['Optimality Gap (%)'], bins=20, color='lightgreen', edgecolor='black')
        plt.xlabel('Optimality Gap (%)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Optimality Gaps')
        plt.show()
        print("\nMean Optimality Gap: {:.2f}%".format(df['Optimality Gap (%)'].mean()))

# =============================================================================
# Main Function
# =============================================================================
def main():
    # Experiment parameters
    NUM_RUNS = 30         # Number of independent runs
    POPULATION_SIZE = 20  # Number of particles in the swarm
    ITERATIONS = 100      # Iterations per run
    NUM_NODES = 30        # Number of cities (nodes) in the TSP instance
    
    # Run experiments and save results to CSV (pso_out.csv)
    df, nodes = run_experiments(num_runs=NUM_RUNS, 
                                population_size=POPULATION_SIZE, 
                                iterations=ITERATIONS, 
                                num_nodes=NUM_NODES, 
                                output_csv='pso_out.csv')
    
    # Analyze the results using the data from pso_out.csv
    # (Set optimal_value to your known best/optimal value if available; here we assume 2000 for example)
    analyze_results(df, optimal_value=2000.0)

if __name__ == '__main__':
    main()
