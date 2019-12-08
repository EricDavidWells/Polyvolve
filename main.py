import numpy as np
import random
from skimage import metrics
import cv2
from matplotlib import pyplot as plt


def init_poly_dna_cv2(polyNum, verticeNum):
    dna = np.random.random(verticeNum*polyNum*2 + polyNum*4)
    return dna


def create_poly_from_dna(dna, polyNum, verticeNum, shape, dtype=np.uint8):

    ptsarray = np.array(dna[0:verticeNum*polyNum*2].reshape([polyNum, verticeNum, 2]))
    colarray = np.array(dna[verticeNum*polyNum*2::].reshape([polyNum, 4]))

    ptsarray[:,:,0] = ptsarray[:,:,0]*shape[1]
    ptsarray[:,:,1] = ptsarray[:,:,1]*shape[0]
    ptsarray = ptsarray.astype(int)

    colarray[:,:3] = colarray[:,:3]*255
    colarray[:,:3] = colarray[:,:3].astype(int)

    img = np.ones([shape[0], shape[1], 3], dtype=dtype)*0  # image is defined this way to preserve the data type of the target image

    for i in range(0, polyNum):
        overlay = img.copy()
        pts = ptsarray[i]
        color = tuple(colarray[i,0:3].tolist())
        alpha = colarray[i, 3]/1.2
        cv2.fillPoly(overlay, [pts], color)
        img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
        # cv2.imshow("check", img)
        # cv2.waitKey()

    # rgbimg = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    # labimg = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img


def rico_ssim(img1, img2, shape):

    temp1 = cv2.resize(img1, shape)
    temp2 = cv2.resize(img2, shape)
    ssim = metrics.structural_similarity(temp1, temp2, multichannel=True)
    return ssim


def rico_mse(image1, image2, displayFlag = False):
    diff = np.subtract(image1, image2)
    err = np.sum(np.square(diff))
    return err


def rico_mse_hsv(image1, image2):
    # print(image1.shape)
    hsv1 = cv2.cvtColor(image1, cv2.COLOR_RGB2HSV)
    hsv2 = cv2.cvtColor(image2, cv2.COLOR_RGB2HSV)
    diff = np.subtract(hsv1, hsv2)
    err = np.sum(np.square(diff))

    return err


def rico_mse_lab(image1, image2):
    lab1 = cv2.cvtColor(image1, cv2.COLOR_RGB2LAB)
    lab2 = cv2.cvtColor(image2, cv2.COLOR_RGB2LAB)
    diff = np.subtract(lab1, lab2)
    err = np.sum(np.square(diff))

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


def randomcrossover(mom, dad):
    child = np.zeros(mom.shape)
    for i in range(0, len(mom)):
        if random.random() < 0.5:
            child[i] = mom[i]
        else:
            child[i] = dad[i]

    return child


def randmutation(child, mutationrate=0.05):

    mutations = 0
    mutant = np.array(child)
    for i in range(0, len(child)):
        if random.random() < mutationrate:
            mutant[i] = random.random()
            mutations += 1

    return mutant, mutations


def randmutation_amount(child, mutationrate, shiftmax):

    mutations = 0
    mutant = np.array(child)
    for i in range(0, len(child)):
        if random.random() < mutationrate:
            mutant[i] += (random.random()-0.5)*shiftmax*2
            mutant[i] = max(mutant[i], 0)
            mutant[i] = min(mutant[i], 1)
            mutations += 1

    return mutant, mutations


def randmutation_amount2(child, mutationrate, mutationamount):
    mutations = 0
    mutant = np.array(child)
    if random.random() < mutationrate:
        for i in range(0, len(child)):
            if random.random() < mutationamount:
                mutant[i] += (random.random() - 0.5) * 0.2 * 2
                mutant[i] = max(mutant[i], 0)
                mutant[i] = min(mutant[i], 1)
                mutant[i] = random.random()
                mutations += 1

    return mutant, mutations


def savepopulation(imgarray, filebase):
    for i in range(0, len(imgarray)):
        cv2.imwrite(filebase + str(i) + ".png", imgarray[i])


if __name__ == "__main__":
    targetimg = cv2.imread("lion.png")
    # targetimg = cv2.imread("monalisa2.png")
    originalshape = targetimg.shape
    displayshape = (300, int(300*targetimg.shape[1]/targetimg.shape[0]))
    calcshape = (25, int(25*targetimg.shape[1]/targetimg.shape[0]))

    calctargetimg = cv2.resize(targetimg, calcshape)
    displaytargetimg = cv2.resize(targetimg, tuple(np.flip(displayshape)))
    cv2.imshow("displaytarget", displaytargetimg)

    # shape = np.shape(targetimg)
    polyNum = 125
    verticeNum = 3
    popNum = 100
    survivorNum = int(popNum*0.1) + 1
    mutationrate = 0.01
    mutationamount = 0.1

    gensperplot = 10
    genstotal = 10000
    fitlog = []
    # plt.ion()
    # plt.show()

    pop = []
    for i in range(0, popNum):
        pop.append(init_poly_dna_cv2(polyNum, verticeNum))

    for h in range(0, int(genstotal/gensperplot)):
        for g in range(0, gensperplot):

            polys = [create_poly_from_dna(x, polyNum, verticeNum, calcshape) for x in pop]
            fits = [rico_ssim(x, calctargetimg, calcshape) for x in polys]
            # fits = [rico_mse_lab(x, targetimg) for x in polys]
            fitind = np.argsort(-np.array(fits))
            survivors = [pop[i] for i in fitind[:survivorNum]]

            while len(survivors) < popNum:
                parent1 = survivors[random.randint(0,survivorNum-1)]
                parent2 = survivors[random.randint(0,survivorNum-1)]
                # child, cross1, cross2 = twopointcrossover(survivors[random.randint(0,survivorNum-1)], survivors[random.randint(0,survivorNum-1)])
                child = randomcrossover(parent1, parent2)
                # mutant, mutationnum = randmutation(child, mutationrate)
                mutant, mutationnum = randmutation_amount(child, mutationrate, mutationamount)
                survivors.append(mutant)

            pop = list(survivors)
            fitlog.append(fits[fitind[0]])

            # print(g)
            displayimg = create_poly_from_dna(pop[fitind[0]], polyNum, verticeNum, displayshape)
            cv2.imshow("rgbimg", displayimg)
            cv2.waitKey(1)

        plt.plot(fitlog, 'b')
        plt.draw()
        plt.pause(0.001)
        savepopulation([displayimg], r"images\fkm8s")
        # input("press enter: ")
        # cv2.imshow("best", create_poly_from_dna(survivors[0], polyNum, verticeNum, shape, np.uint8))
        # cv2.waitKey()

    cv2.waitKey()
# need to do roulette selection for survivors or something, convert images to hsv at beginning, add alpha channel using this
# https://gist.github.com/IAmSuyogJadhav/305bfd9a0605a4c096383408bee7fd5c

