import DrawingFunctions
import numpy as np
import random
from skimage import metrics
from deap import base
from deap import creator
from deap import tools
import cv2


def init_poly_dna_cv2(polyNum, verticeNum):
    dna = np.random.random(verticeNum*polyNum*2 + polyNum*3)
    return dna


def create_poly_from_dna(dna, polyNum, verticeNum, shape, dtype=np.int8):

    ptsarray = np.array(dna[0:verticeNum*polyNum*2].reshape([polyNum, verticeNum, 2]))
    colarray = np.array(dna[verticeNum*polyNum*2::].reshape([polyNum, 3]))

    ptsarray[:,:,0] = ptsarray[:,:,0]*shape[1]
    ptsarray[:,:,1] = ptsarray[:,:,1]*shape[0]
    colarray = colarray*255

    ptsarray = ptsarray.astype(int)
    colarray = colarray.astype(int)

    img = np.zeros(shape, dtype=dtype)  # image is defined this way to preserve the data type of the target image

    for i in range(0, polyNum):
        pts = ptsarray[i]
        color = tuple(colarray[i].tolist())
        cv2.fillPoly(img, [pts], color)

    return img


def rico_ssim(img1, img2):
    ssim = metrics.structural_similarity(img1, img2, multichannel=True)
    return ssim


def rico_mse(image1, image2):
    # err = measure.compare_mse(image1, image2)

    rgbimg1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    # hsv2 = cv2.cvtColor(image2, cv2.COLOR_RGB2HSV)
    # err = ((image1-image2)**2).mean()
    err = ((rgbimg1-image2)**2).mean()
    return err

def twopointcrossover(mom, dad):
    """
    :param mom: first parent chromosome
        type: list of numbers
    :param dad: second parent chromosome
        type: list of numbers
    :return: child: chromosome made from both parents using a two point crossover method
             cross1, cross2: crossover points used for the two point crossover method

    """

    r1 = random.randint(0, len(mom)-1)
    r2 = random.randint(0, len(dad)-1)

    cross1 = min(r1, r2)
    cross2 = max(r1, r2)

    child = np.array(mom[0:cross1])
    child = np.append(child, dad[cross1:cross2])
    child = np.append(child, mom[cross2::])

    return child, cross1, cross2


def randmutation(child, mutationrate=0.05, minvalue=0, maxvalue=1000):

    mutations = 0
    mutant = np.array(child)
    for i in range(0, len(child)):
        if random.random() < mutationrate:
            mutant[i] = random.random()*(maxvalue-minvalue) + minvalue
            mutations += 1

    return mutant, mutations


if __name__ == "__main__":
    targetimg = cv2.imread("IMG_2497.jpg")
    shape = np.shape(targetimg)
    polyNum = 100
    verticeNum = 4
    popNum = 50
    survivorNum = 10
    mutationrate = 0.01

    # dna1 = init_poly_dna_cv2(polyNum, verticeNum)
    # img1 = create_poly_from_dna(dna1, polyNum, verticeNum, shape, np.uint8)
    #
    # dna2 = init_poly_dna_cv2(polyNum, verticeNum)
    # img2 = create_poly_from_dna(dna2, polyNum, verticeNum, shape, np.uint8)
    #
    # dna3, cross1, cross2 = twopointcrossover(dna1, dna2)
    # img3 = create_poly_from_dna(dna3, polyNum, verticeNum, shape, np.uint8)
    #
    # dna4, mutationnum = randmutation(dna3, 0.025, min(shape))
    # img4 = create_poly_from_dna(dna4, polyNum, verticeNum, shape, np.uint8)
    #
    # cv2.imshow("target", targetimg)
    # cv2.imshow("img1", img1)
    # cv2.imshow("img2", img2)
    # cv2.imshow("img3", img3)
    # cv2.imshow("img4", img4)
    # cv2.waitKey()

    pop = []
    for i in range(0, popNum):
        pop.append(init_poly_dna_cv2(polyNum, verticeNum))

    for h in range(0, 10):
        for g in range(0, 1000):

            fits = list(map(lambda x: rico_mse(create_poly_from_dna(x, polyNum, verticeNum, shape, np.uint8), targetimg), pop))
            fitind = np.argsort(np.array(fits))
            # survivors = pop[fitind[:survivorNum]]
            survivors = [pop[i] for i in fitind[:survivorNum]]

            while len(survivors) < popNum:
                child, cross1, cross2 = twopointcrossover(survivors[random.randint(0,4)], survivors[random.randint(0,4)])
                mutant, mutationnum = randmutation(child, mutationrate, 0, 1)
                survivors.append(mutant)

            print(fits[fitind[0]], h, g)
            pop = list(survivors)

        cv2.imshow("best", create_poly_from_dna(survivors[0], polyNum, verticeNum, shape, np.uint8))
        cv2.waitKey()


# need to do roulette selection for survivors or something, convert images to hsv at beginning, add alpha channel using this
# https://gist.github.com/IAmSuyogJadhav/305bfd9a0605a4c096383408bee7fd5c

