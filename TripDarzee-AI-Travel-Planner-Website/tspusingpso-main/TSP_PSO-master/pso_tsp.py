import math
import random
import time as tm
from tqdm import tqdm
from matplotlib import pyplot as plt


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
        # Generate random particles
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

                # Apply swaps to the original route with probabilities
                new_route = particle.route.copy()
                for swap in temp_velocity:
                    if random.random() <= swap[2]:
                        new_route[swap[0]], new_route[swap[1]] = new_route[swap[1]], new_route[swap[0]]
                particle.route = new_route
                # Calculate new cost
                particle.current_cost = self._calculate_distance(new_route)
                particle.update_pbest()
                # Update global best
                if particle.pbest_cost < self.gbest_distance:
                    self.gbest = particle.pbest.copy()
                    self.gbest_distance = particle.pbest_cost

        runtime = tm.time() - start_time
        return runtime, self.gbest_distance

    def plot(self, line_width=1, point_radius=math.sqrt(2.0), annotation_size=8, dpi=120, save=True, name=None):
        if self.gbest is None:
            print("No route to plot.")
            return
        x = [self.nodes[i][0] for i in self.gbest]
        x.append(x[0])
        y = [self.nodes[i][1] for i in self.gbest]
        y.append(y[0])
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, linewidth=line_width, marker='o', markersize=point_radius * 10)
        plt.scatter(x, y, s=math.pi * (point_radius ** 2.0) * 100)
        plt.title(f'PSO - Tour Distance: {round(self.gbest_distance, 2)}')
        for i in self.gbest:
            plt.annotate(self.labels[i], self.nodes[i], size=annotation_size)
        if save:
            name = f'pso_tour.png' if name is None else name
            plt.savefig(name, dpi=dpi)
        plt.show()
        plt.close()


if __name__ == '__main__':
    # Test 1: 4-node square (Optimal distance = 4.0)
    print("\nTesting 4-node square...")
    nodes_square = [(0, 0), (0, 1), (1, 1), (1, 0)]
    pso = SolveTSPUsingPSO(population_size=10, iterations=100, pbest_prob=0.9, gbest_prob=0.1, nodes=nodes_square)
    runtime, distance = pso.run()
    print(f"PSO Best Distance: {distance} (Expected: ~4.0)")
    pso.plot()

    # Test 2: 3-node triangle (Optimal distance ≈ 3.414)
    print("\nTesting 3-node triangle...")
    nodes_triangle = [(0, 0), (0, 1), (1, 0)]
    pso = SolveTSPUsingPSO(population_size=10, iterations=100, pbest_prob=0.9, gbest_prob=0.1, nodes=nodes_triangle)
    runtime, distance = pso.run()
    print(f"PSO Best Distance: {distance} (Expected: ~3.414)")
    pso.plot()

    # Enhanced experiments
    with open('./pso_out.csv', 'w') as f:
        f.write('Iteration,Population,Time,Distance\n')
        population_size = 20
        iterations = 200

        for i in range(50):
            print(f'Iteration: {i + 1}')
            for j in tqdm(range(20), desc='Trials'):
                nodes = [(random.uniform(-400, 400), random.uniform(-400, 400)) for _ in range(10 * (i + 1))]
                pso = SolveTSPUsingPSO(population_size=population_size, iterations=iterations,
                                      pbest_prob=0.9, gbest_prob=0.1, nodes=nodes)
                time, dist = pso.run()
                f.write(f'{i + 1},{population_size},{time},{dist}\n')
                f.flush()