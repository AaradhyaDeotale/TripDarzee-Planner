class Particle {
    constructor(route, cost) {
      this.route = [...route];
      this.pbest = [...route];
      this.current_cost = cost;
      this.pbest_cost = cost;
      this.velocity = [];
    }
  
    clearVelocity() {
      this.velocity = [];
    }
  
    updatePBest() {
      if (this.current_cost < this.pbest_cost) {
        this.pbest = [...this.route];
        this.pbest_cost = this.current_cost;
      }
    }
  }
  
  export class SolveTSPUsingPSO {
    constructor({ nodes, populationSize = 20, iterations = 100, pbestProb = 0.9, gbestProb = 0.1 }) {
      this.nodes = nodes;
      this.populationSize = populationSize;
      this.iterations = iterations;
      this.pbestProb = pbestProb;
      this.gbestProb = gbestProb;
      this.numNodes = nodes.length;
      this.particles = [];
      this.gbest = null;
      this.gbestDistance = Infinity;
      this.initializeParticles();
    }
  
    initializeParticles() {
      // Generate random particles
      for (let i = 0; i < this.populationSize - 1; i++) {
        const route = this.shuffle([...Array(this.numNodes).keys()]);
        const distance = this.calculateDistance(route);
        this.particles.push(new Particle(route, distance));
      }
      
      // Generate greedy route particle
      const greedyRoute = this.greedyRoute();
      const greedyDistance = this.calculateDistance(greedyRoute);
      this.particles.push(new Particle(greedyRoute, greedyDistance));
      
      // Initialize global best
      this.gbest = [...this.particles[0].pbest];
      this.gbestDistance = this.particles[0].pbest_cost;
      this.updateGlobalBest();
    }
  
    shuffle(array) {
      for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
      }
      return array;
    }
  
    greedyRoute(startIndex = 0) {
      const unvisited = [...Array(this.numNodes).keys()];
      const route = [];
      let current = startIndex;
      route.push(current);
      unvisited.splice(unvisited.indexOf(current), 1);
  
      while (unvisited.length > 0) {
        let nearest = unvisited[0];
        let minDist = this.distance(current, nearest);
        for (const node of unvisited) {
          const d = this.distance(current, node);
          if (d < minDist) {
            minDist = d;
            nearest = node;
          }
        }
        route.push(nearest);
        current = nearest;
        unvisited.splice(unvisited.indexOf(nearest), 1);
      }
      return route;
    }
  
    distance(a, b) {
      const dx = this.nodes[a][0] - this.nodes[b][0];
      const dy = this.nodes[a][1] - this.nodes[b][1];
      return Math.sqrt(dx ** 2 + dy ** 2);
    }
  
    calculateDistance(route) {
      let total = 0;
      for (let i = 0; i < route.length; i++) {
        total += this.distance(route[i], route[(i + 1) % route.length]);
      }
      return total;
    }
  
    updateGlobalBest() {
      for (const particle of this.particles) {
        if (particle.pbest_cost < this.gbestDistance) {
          this.gbest = [...particle.pbest];
          this.gbestDistance = particle.pbest_cost;
        }
      }
    }
  
    async run() {
      const startTime = Date.now();
      for (let iter = 0; iter < this.iterations; iter++) {
        for (const particle of this.particles) {
          particle.clearVelocity();
          const newRoute = [...particle.route];
          let changed = false;
  
          // Apply pbest swaps
          for (let i = 0; i < newRoute.length; i++) {
            if (newRoute[i] !== particle.pbest[i]) {
              const j = particle.pbest.indexOf(newRoute[i]);
              if (Math.random() <= this.pbestProb) {
                [newRoute[i], newRoute[j]] = [newRoute[j], newRoute[i]];
                changed = true;
              }
            }
          }
  
          // Apply gbest swaps
          for (let i = 0; i < newRoute.length; i++) {
            if (newRoute[i] !== this.gbest[i]) {
              const j = this.gbest.indexOf(newRoute[i]);
              if (Math.random() <= this.gbestProb) {
                [newRoute[i], newRoute[j]] = [newRoute[j], newRoute[i]];
                changed = true;
              }
            }
          }
  
          if (changed) {
            particle.route = [...newRoute];
            particle.current_cost = this.calculateDistance(newRoute);
            particle.updatePBest();
          }
        }
        this.updateGlobalBest();
      }
      return { distance: this.gbestDistance, route: this.gbest };
    }
  }