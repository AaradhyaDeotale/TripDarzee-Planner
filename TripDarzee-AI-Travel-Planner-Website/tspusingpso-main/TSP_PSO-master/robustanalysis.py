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
        # Basic run (without convergence tracking) used for quick tests
        start_time = tm.time()
        for _ in range(self.iterations):
            for particle in self.particles:
                particle.clear_velocity()
                temp_velocity = []
                pbest_route = particle.pbest
                gbest_route = self.gbest
                new_route = particle.route.copy()
                # Generate swaps toward personal best
                for i in range(len(new_route)):
                    if new_route[i] != pbest_route[i]:
                        j = pbest_route.index(new_route[i])
                        swap = (i, j, self.pbest_prob)
                        temp_velocity.append(swap)
                        new_route[i], new_route[j] = new_route[j], new_route[i]
                # Generate swaps toward global best
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
                # Generate swaps toward personal best
                for i in range(len(new_route)):
                    if new_route[i] != pbest_route[i]:
                        j = pbest_route.index(new_route[i])
                        swap = (i, j, self.pbest_prob)
                        temp_velocity.append(swap)
                        new_route[i], new_route[j] = new_route[j], new_route[i]
                # Generate swaps toward global best
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
def run_experiments(num_runs=30, population_size=20, iterations=100, num_nodes=30, output_csv='pso_result.csv'):
    """
    Run multiple independent runs on a fixed TSP instance.
    For each run, record per-iteration best distance, total runtime, and add a run ID.
    Save all data to output_csv.
    """
    results = []
    # Create a fixed TSP instance (same nodes for all runs)
    nodes = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_nodes)]
    
    for run in tqdm(range(1, num_runs + 1), desc='Experiment Runs'):
        pso = SolveTSPUsingPSOConvergence(population_size=population_size,
                                          iterations=iterations,
                                          pbest_prob=0.9,
                                          gbest_prob=0.1,
                                          nodes=nodes)
        run_runtime, best_distance, convergence_curve = pso.run()
        # Record each iteration's best distance along with the run ID
        for iteration, distance in enumerate(convergence_curve, start=1):
            results.append({
                'Run': run,
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

def analyze_results(df, optimal_value=None, success_threshold=5.0):
    """
    Analyze the CSV data and produce tangible measures:
    - Overall statistics and consistency (mean, std, etc.)
    - Histograms of tour distances and runtimes
    - Convergence curve (mean and std per iteration)
    - Optimality gap analysis and success rate (if optimal_value provided)
    """
    print("=== Overall Statistics ===")
    print(df.describe())
    
    # Histogram: Tour Distances (all iterations)
    plt.figure(figsize=(10, 6))
    plt.hist(df['Distance'], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Tour Distance')
    plt.ylabel('Frequency')
    plt.title('Histogram of PSO Tour Distances (All Iterations)')
    plt.show()
    
    # Histogram: Runtimes (per run; note these are repeated per run, so we take unique values per run)
    runtimes = df.groupby('Run')['Time'].first().values
    plt.figure(figsize=(10, 6))
    plt.hist(runtimes, bins=20, color='salmon', edgecolor='black')
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
    
    if optimal_value is not None:
        # Calculate optimality gap (%) for each measurement
        df['Optimality Gap (%)'] = (df['Distance'] - optimal_value) / optimal_value * 100
        
        plt.figure(figsize=(10, 6))
        plt.hist(df['Optimality Gap (%)'], bins=20, color='lightgreen', edgecolor='black')
        plt.xlabel('Optimality Gap (%)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Optimality Gaps (All Iterations)')
        plt.show()
        
        print("\nMean Optimality Gap: {:.2f}%".format(df['Optimality Gap (%)'].mean()))
        
        # Compute success rate based on final iteration of each run
        final_iter = df.groupby('Run').last().reset_index()
        # Success if the final optimality gap is within the success_threshold (%)
        success_count = (final_iter['Distance'] <= optimal_value * (1 + success_threshold/100)).sum()
        success_rate = success_count / len(final_iter) * 100
        print("Success Rate (final run within {:.1f}% of optimal): {:.2f}%".format(success_threshold, success_rate))

def print_additional_statistics(df):
    """
    Print additional statistical measures such as mean, std, min, max for final iteration distances and runtimes.
    """
    final_iter = df.groupby('Run').last().reset_index()
    print("\n=== Final Iteration Statistics per Run ===")
    print("Mean Final Tour Distance: {:.2f}".format(final_iter['Distance'].mean()))
    print("Std Dev Final Tour Distance: {:.2f}".format(final_iter['Distance'].std()))
    print("Min Final Tour Distance: {:.2f}".format(final_iter['Distance'].min()))
    print("Max Final Tour Distance: {:.2f}".format(final_iter['Distance'].max()))
    
    runtimes = df.groupby('Run')['Time'].first().reset_index()['Time']
    print("\nMean Runtime per Run: {:.2f} s".format(runtimes.mean()))
    print("Std Dev Runtime per Run: {:.2f} s".format(runtimes.std()))

# =============================================================================
# Main Function
# =============================================================================
def main():
    # Experiment parameters
    NUM_RUNS = 30         # Number of independent runs
    POPULATION_SIZE = 20  # Number of particles in the swarm
    ITERATIONS = 100      # Iterations per run
    NUM_NODES = 30        # Number of cities (nodes) in the TSP instance
    OUTPUT_CSV = 'pso_result.csv'
    
    # Run experiments and save results to CSV
    df, nodes = run_experiments(num_runs=NUM_RUNS, 
                                population_size=POPULATION_SIZE, 
                                iterations=ITERATIONS, 
                                num_nodes=NUM_NODES, 
                                output_csv=OUTPUT_CSV)
    
    # Analyze the results using the data from CSV.
    # Set optimal_value to your known best/optimal value for the TSP instance (if available).
    KNOWN_OPTIMAL = 2000.0  # Adjust this as needed or set to None if unknown
    analyze_results(df, optimal_value=KNOWN_OPTIMAL, success_threshold=5.0)
    
    print_additional_statistics(df)

if __name__ == '__main__':
    main()
