import random

import numpy as np


def two_point_crossover(mom, dad):
    """
    :param mom: first parent chromosome
        type: list of numbers
    :param dad: second parent chromosome
        type: list of numbers
    :return: child: chromosome made from both parents using a two point crossover method
             cross1, cross2: crossover points used for the two point crossover method

    """

    r1 = random.randint(0, len(mom) - 1)
    r2 = random.randint(0, len(dad) - 1)

    cross1 = min(r1, r2)
    cross2 = max(r1, r2)

    child = np.array(mom[0:cross1])
    child = np.append(child, dad[cross1:cross2])
    child = np.append(child, mom[cross2::])

    return child


def random_crossover(mom, dad):
    child = np.zeros(mom.shape)
    for i in range(len(mom)):
        if random.random() < 0.5:
            child[i] = mom[i]
        else:
            child[i] = dad[i]

    return child


def random_mutation(child, mutation_rate=0.05):
    """
    Every dna item has a mutation_rate chance of becoming a random value
    """
    mutations = 0
    mutant = np.array(child)
    for i in range(len(child)):
        if random.random() < mutation_rate:
            mutant[i] = random.random()
            mutations += 1

    return mutant, mutations


def random_mutation_shift(child, mutation_rate, max_shift):
    """
    Every dna item has a mutation_rate chance of incrementing by a random percentage of max_shift in either direction
    """
    mutations = 0
    mutant = np.array(child)
    for i in range(len(child)):
        if random.random() < mutation_rate:
            mutant[i] += random.uniform(-1, 1) * max_shift
            mutant[i] = max(mutant[i], 0)
            mutant[i] = min(mutant[i], 1)
            mutations += 1

    return mutant, mutations


def random_mutation_single(child, mutation_rate, mutation_amount):
    """
    Every child has a mutation_rate chance of becoming a mutant:  if it is than each piece of dna has a mutation_amount
    chance of being randomized.
    """
    mutations = 0
    mutant = np.array(child)
    if random.random() < mutation_rate:
        for i in range(len(child)):
            if random.random() < mutation_amount:
                mutant[i] = random.random()
                mutations += 1

    return mutant, mutations
