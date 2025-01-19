import numpy as np


class Population:
    def __init__(
        self,
        pop_size: int,
        dna_cnt: int,
        dna_len: int,
        mutation_rate: float,
        crossover_rate: float,
        elitism_rate: float,
        lower_bound: int,
        upper_bound: int,
    ) -> None:
        self.pop_size = pop_size
        self.dna_cnt = dna_cnt
        self.dna_len = dna_len
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.populations = self._init_population(pop_size, dna_cnt, dna_len)
        self.elites = None

    def _init_population(self, pop_size, dna_cnt, dna_len) -> np.ndarray:
        populations = np.random.randint(2, size=(pop_size, dna_cnt, dna_len))
        return populations

    def decodeDNA(self, pops=None):
        if pops is None:
            pops = self.populations
        weights = 2 ** np.arange(pops.shape[-1])
        max_value = 2 ** pops.shape[-1] - 1
        decoded = (pops * weights).sum(axis=-1)
        scaled = self.lower_bound + (decoded / max_value) * (
            self.upper_bound - self.lower_bound
        )
        return scaled

    def _encodeDNA(self, decoded_pops=np.array):
        adjusted_pops = np.round(decoded_pops - self.lower_bound).astype(int)
        encoded = [
            [list(map(int, bin(num)[2:].zfill(self.dna_len))) for num in row]
            for row in adjusted_pops
        ]
        return np.array(encoded)[:, :, ::-1]

    def mutate(self):
        random_numbers = np.random.random(self.populations.shape)
        mask = random_numbers < self.mutation_rate
        self.populations[mask] = 1 - self.populations[mask]
        self._preserve_elites()

    def crossover(self):
        shuffled_indices = np.random.permutation(self.pop_size)
        next_pops = self.populations[shuffled_indices].copy()
        for i in range(0, self.pop_size - 1, 2):
            if np.random.rand() < self.crossover_rate:
                parent1, parent2 = next_pops[i], next_pops[i + 1]
                cross_point = np.random.randint(1, self.dna_len)
                parent1[:, cross_point:], parent2[:, cross_point:] = (
                    parent2[:, cross_point:].copy(),
                    parent1[:, cross_point:].copy(),
                )
        self.populations = next_pops
        self._preserve_elites()

    def select(self, fitness):
        if fitness.sum() == 0:
            return

        # Select elite individuals
        num_elites = max(1, int(self.pop_size * self.elitism_rate))
        elite_indices = np.argsort(fitness)[-num_elites:]
        self.elites = self.populations[elite_indices]

        # Select the remaining individuals
        probabilities = fitness / fitness.sum()
        selected_indices = np.random.choice(
            np.arange(self.pop_size),
            size=self.pop_size - num_elites,
            replace=True,
            p=probabilities,
        )
        selected = self.populations[selected_indices]

        # Combine elites and selected individuals
        self.populations = np.vstack((self.elites, selected))

    def _preserve_elites(self):
        num_elites = len(self.elites)
        self.populations[:num_elites] = self.elites
