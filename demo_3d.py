import numpy as np
import matplotlib.pyplot as plt
from GeneticAlgorithm import Population

np.random.seed(0)


# Rastrigin Function
def F(x):
    return 10 * x.shape[-1] + np.sum(
        x**2 - 10 * np.cos(2 * np.pi * x), axis=-1
    )


# Parameters
pop_size = 20
dna_cnt = 2
dna_len = 10
mutation_rate = 0.02
crossover_rate = 0.8
elitism_rate = 0.05
lower_bound = -3
upper_bound = 3
max_generations = 200

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

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")

x = np.linspace(lower_bound, upper_bound, 100)
y = np.linspace(lower_bound, upper_bound, 100)
X, Y = np.meshgrid(x, y)
Z = F(np.stack([X, Y], axis=-1))

ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.2)
ax.set_title("3D Rastrigin Function")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("F(x, y)")

for generation in range(1, max_generations + 1):
    decoded = population.decodeDNA()
    fitness = F(decoded).flatten()

    fitness_ratio = 1 / (1 + fitness)

    best_values.append(fitness_ratio.max())
    mean_values.append(fitness_ratio.mean())

    ax.collections.clear()
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.2)
    decoded_x = decoded[:, 0]
    decoded_y = decoded[:, 1]
    decoded_z = F(decoded)
    ax.scatter(
        decoded_x, decoded_y, decoded_z, color="red", label="Population"
    )
    ax.set_title(f"Generation {generation}")
    ax.legend()
    plt.pause(0.1)

    population.select(fitness_ratio)
    population.crossover()
    population.mutate()

# Plot the change in best and mean fitness
plt.figure()
plt.plot(
    range(max_generations), best_values, label="Best Fitness", color="green"
)
plt.plot(
    range(max_generations), mean_values, label="Mean Fitness", color="orange"
)
plt.title("Fitness Convergence")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.show()
