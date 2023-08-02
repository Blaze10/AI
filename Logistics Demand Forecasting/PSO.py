import random
import numpy as np
from DemandPrediction import DemandPrediction
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.best_position = position
        self.best_score = np.inf

class PSO:
    def __init__(self, num_particles, num_iterations, dataset_name):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.dataset_name = dataset_name
        self.bounds = DemandPrediction.bounds()
        self.particles = [Particle(self.random_position(), self.random_velocity()) for i in range(num_particles)]
        self.best_global_position = None
        self.best_global_score = np.inf

    def random_position(self):
        return [random.uniform(b[0], b[1]) for b in self.bounds]

    def random_velocity(self):
        return [random.uniform(-1, 1) for i in range(DemandPrediction.N_PARAMETERS)]

    def run(self):
        scores = []
        for i in range(self.num_iterations):
            scores.append(self.best_global_score)
            for particle in self.particles:
                if not DemandPrediction.is_valid(particle.position):
                    particle.position = self.random_position()
                if not DemandPrediction.is_valid(particle.velocity):
                    particle.velocity = self.random_velocity()

                score = DemandPrediction(self.dataset_name).evaluate(particle.position)
                if score < particle.best_score:
                    particle.best_position = particle.position
                    particle.best_score = score

                if score < self.best_global_score:
                    self.best_global_position = particle.position
                    self.best_global_score = score

                for j in range(DemandPrediction.N_PARAMETERS):
                    r1 = random.uniform(0, 1)
                    r2 = random.uniform(0, 1)
                    cognitive = 2.0
                    social = 2.0
                    velocity = (cognitive * r1 * (particle.best_position[j] - particle.position[j]) +
                                social * r2 * (self.best_global_position[j] - particle.position[j]))
                    particle.velocity[j] = velocity
                    particle.position[j] += velocity

        plt.plot(scores)
        plt.title('PSO optimization')
        plt.xlabel('Iteration')
        plt.ylabel('Best Global Score')
        plt.show()

        return self.best_global_position, self.best_global_score

pso = PSO(num_particles=50, num_iterations=100, dataset_name="train")
best_params, best_score = pso.run()
print("Best parameters found:", best_params)
print("Best score found:", best_score)



