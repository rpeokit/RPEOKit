import numpy as np
import random

class GWO:
    def __init__(self, objective_function, dim, bounds, num_wolves=5, max_iter=10):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = bounds
        self.num_wolves = num_wolves
        self.max_iter = max_iter

        self.wolves = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.num_wolves, self.dim))
        self.alpha_pos = np.zeros(self.dim)
        self.alpha_score = float("inf")

        self.beta_pos = np.zeros(self.dim)
        self.beta_score = float("inf")

        self.delta_pos = np.zeros(self.dim)
        self.delta_score = float("inf")

    def optimize(self):
        for iter in range(self.max_iter):
            for i in range(self.num_wolves):
                fitness = self.objective_function(self.wolves[i])

                if fitness < self.alpha_score:
                    self.alpha_score = fitness
                    self.alpha_pos = self.wolves[i].copy()

                elif fitness < self.beta_score:
                    self.beta_score = fitness
                    self.beta_pos = self.wolves[i].copy()

                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.wolves[i].copy()

            a = 2 - iter * (2 / self.max_iter)

            for i in range(self.num_wolves):
                for j in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1, C1 = 2 * a * r1 - a, 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.wolves[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    A2, C2 = 2 * a * r1 - a, 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - self.wolves[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3, C3 = 2 * a * r1 - a, 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - self.wolves[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta

                    # 更新位置，确保在边界范围内
                    self.wolves[i, j] = np.clip((X1 + X2 + X3) / 3, self.bounds[j, 0], self.bounds[j, 1])

            print(f"Iteration {iter + 1}/{self.max_iter}, Alpha Score: {self.alpha_score}")

        return self.alpha_pos, self.alpha_score
