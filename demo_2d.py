import numpy as np
import matplotlib.pyplot as plt
from GeneticAlgorithm import Population

np.random.seed(0)


# Rastrigin Function
def F(x):
    return x**2 - 10 * np.cos(2 * np.pi * x) + 10


# Parameters
pop_size = 20
dna_cnt = 1
dna_len = 10
mutation_rate = 0.02
crossover_rate = 0.8
elitism_rate = 0.05
lower_bound = -6
upper_bound = 6
max_generations = 100

population = Population(
    pop_size=pop_size,
    dna_cnt=dna_cnt,
    dna_len=dna_len,
    mutation_rate=mutation_rate,
    crossover_rate=crossover_rate,
    elitism_rate=elitism_rate,
    lower_bound=lower_bound,
    upper_bound=upper_bound,
)

best_values = []
mean_values = []

plt.figure()
x = np.linspace(lower_bound, upper_bound, int(upper_bound - lower_bound) * 100)
y = F(x)

for generation in range(1, max_generations + 1):
    decoded = population.decodeDNA()
    fitness = F(decoded).flatten()

    fitness_ratio = 1 / (1 + fitness)

    best_values.append(fitness_ratio.max())
    mean_values.append(fitness_ratio.mean())

    plt.clf()
    plt.plot(x, y, label="F(x)", color="blue")
    plt.scatter(
        decoded, F(decoded), color="red", label="Population", alpha=0.6
    )
    plt.title(f"Generation {generation}")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.legend()
    plt.pause(0.1)
    population.select(fitness_ratio)
    population.crossover()
    population.mutate()


# Plot the change in best and mean fitness
plt.figure()
plt.plot(
    range(max_generations),
    best_values,
    label="Best Fitness",
    color="green",
)
plt.plot(
    range(max_generations),
    mean_values,
    label="Mean Fitness",
    color="orange",
)
plt.title("Fitness Convergence")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.show()
