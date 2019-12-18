import random

import cv2
import numpy as np


class Population:

    def __init__(self, population_size, init_fn, fitness_fn, crossover_fn, mutation_fn, p_survive, p_reproduce,
                 minimize_fitness=False):

        self.init_fn = init_fn
        self.population = [init_fn() for _ in range(0, population_size)]
        self.population_size = population_size
        self.fitness_fn = fitness_fn
        self.crossover_fn = crossover_fn
        self.mutation_fn = mutation_fn
        self.num_survivors = int(p_survive * population_size)
        # The cutoff point past for individuals to reproduce
        self.num_parents = int(p_reproduce * population_size)
        self.minimize_fitness = minimize_fitness

    @staticmethod
    def evaluate_individual(evaluation_queue, done_queue, fitness_fn):
        not_done = True
        while not_done:
            individual = evaluation_queue.get()
            if individual is None:
                not_done = False
            else:
                if individual.fitness is None:
                    individual.fitness = fitness_fn(individual)
                done_queue.put(1)

    def evaluate(self):
        # Survivors do not need to be evaluated.
        for individual in self.population:
            if individual.fitness is None:
                individual.fitness = self.fitness_fn(individual)

        self.population.sort(key=lambda x: x.fitness, reverse=not self.minimize_fitness)

    def breed(self):
        child_dna = [None] * (self.population_size - self.num_survivors)
        for i in range(self.num_survivors, self.population_size):
            # 2 parents produce one child. We choose these parents from the fittest individuals.
            parent1, parent2 = random.choices(self.population[0:self.num_parents], k=2)
            dna = self.crossover_fn(parent1.dna, parent2.dna)
            dna = self.mutation_fn(dna)
            child_dna[i-self.num_survivors] = dna

        for i in range(self.num_survivors, self.population_size):
            self.population[i].clear()
            self.population[i].dna = child_dna[i - self.num_survivors]

    def display(self, block=True):
        for i, individual in enumerate(self.population):
            individual.display(str(i), block)


class IndividualPoly:
    def __init__(self, num_polygons, num_vertices):
        self.num_polygons = num_polygons
        self.num_vertices = num_vertices
        self.fitness = None
        self.img = None
        self.dna = None

    def randomize(self):
        self.dna = np.random.random(self.num_vertices * self.num_polygons * 2 + self.num_polygons * 4)

    def zero(self):
        self.dna = np.zeros(self.num_vertices * self.num_polygons * 2 + self.num_polygons * 4)

    def zerocolonly(self):
        self.dna = np.random.random(self.num_vertices * self.num_polygons * 2 + self.num_polygons * 4)
        self.dna[self.num_vertices * self.num_polygons * 2::] = 0

    def draw(self, shape, dtype=np.uint8, colorspace="RGB"):

        ptsarray = np.array(self.dna[0:self.num_vertices * self.num_polygons * 2].reshape([self.num_polygons, self.num_vertices, 2]))
        colarray = np.array(self.dna[self.num_vertices * self.num_polygons * 2::].reshape([self.num_polygons, 4]))

        ptsarray[:, :, 0] = ptsarray[:, :, 0] * shape[0]
        ptsarray[:, :, 1] = ptsarray[:, :, 1] * shape[1]
        ptsarray = ptsarray.astype(int)

        colarray[:, :3] = colarray[:, :3] * 255
        colarray[:, :3] = colarray[:, :3].astype(int)

        img = np.zeros((shape[1], shape[0], 3), dtype=dtype)

        for i in range(self.num_polygons):
            overlay = img.copy()
            pts = ptsarray[i]
            color = tuple(colarray[i, 0:3].tolist())
            alpha = colarray[i, 3] / 1.2
            cv2.fillPoly(overlay, [pts], color)
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        if colorspace == "HSV":
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        elif colorspace == "LAB":
            img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

        self.img = img

    def display(self, window_name, block=True):
        cv2.imshow(window_name, self.img)
        if block:
            cv2.waitKey()
        else:
            cv2.waitKey(1)

    def clear(self):
        self.fitness = None
        self.img = None
        self.dna = None

    def save(self, filename):
        cv2.imwrite(filename, self.img)


class IndividualRectangle(IndividualPoly):
    def __init__(self, polynum):
        IndividualPoly.__init__(self, polynum, 2)

    def draw(self, shape, dtype=np.uint8, colorspace="RGB"):
        ptsarray = np.array(self.dna[0:self.num_vertices * self.num_polygons * 2].reshape([self.num_polygons, self.num_vertices, 2]))
        colarray = np.array(self.dna[self.num_vertices * self.num_polygons * 2::].reshape([self.num_polygons, 4]))

        ptsarray[:, :, 0] = ptsarray[:, :, 0] * shape[0]
        ptsarray[:, :, 1] = ptsarray[:, :, 1] * shape[1]
        ptsarray = ptsarray.astype(int)

        colarray[:, :3] = colarray[:, :3] * 255
        colarray[:, :3] = colarray[:, :3].astype(int)

        img = np.zeros([shape[1], shape[0], 3], dtype=dtype)

        for i in range(0, self.num_polygons):
            overlay = img.copy()
            pts = ptsarray[i]
            color = tuple(colarray[i, 0:3].tolist())
            alpha = colarray[i, 3] / 1.2
            cv2.rectangle(overlay, tuple(pts[0]), tuple(pts[1]), color, -1)
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        if colorspace == "HSV":
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        elif colorspace == "LAB":
            img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

        self.img = img


class IndividualCircle(IndividualPoly):
    def __init__(self, polynum):
        IndividualPoly.__init__(self, polynum, 2)

    def draw(self, shape, dtype=np.uint8, colorspace="RGB"):
        ptsarray = np.array(self.dna[0:self.num_vertices * self.num_polygons * 2].reshape([self.num_polygons, self.num_vertices, 2]))
        colarray = np.array(self.dna[self.num_vertices * self.num_polygons * 2::].reshape([self.num_polygons, 4]))

        ptsarray[:, :, 0] = ptsarray[:, :, 0] * shape[0]
        ptsarray[:, :, 1] = ptsarray[:, :, 1] * shape[1]
        ptsarray = ptsarray.astype(int)

        colarray[:, :3] = colarray[:, :3] * 255
        colarray[:, :3] = colarray[:, :3].astype(int)

        img = np.zeros([shape[1], shape[0], 3], dtype=dtype)

        for i in range(0, self.num_polygons):
            overlay = img.copy()
            pts = ptsarray[i]
            color = tuple(colarray[i, 0:3].tolist())
            alpha = colarray[i, 3] / 1.2
            cv2.circle(overlay, tuple(pts[0]), int(pts[1, 0] / 2), color, -1)
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        if colorspace == "HSV":
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        elif colorspace == "LAB":
            img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

        self.img = img
