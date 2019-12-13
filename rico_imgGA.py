import numpy as np
import random
from skimage import metrics
import cv2


class Population:

    def __init__(self, popnum, initfun, fitnessfun, crossoverfun, mutationfun, survivoramt, parentamt, evaltype=-1):

        self.initfun = initfun
        self.pop = [initfun() for _ in range(0, popnum)]
        self.popnum = popnum
        self.fitnessfun = fitnessfun
        self.crossoverfun = crossoverfun
        self.mutationfun = mutationfun
        self.survivornum = int(survivoramt*popnum)
        self.parentnum = int(parentamt*popnum)
        self.evaltype = evaltype

    def evaluate(self):
        for i, individual in enumerate(self.pop):
            if not individual.fitness:
                individual.fitness = self.fitnessfun(individual)

        if self.evaltype == -1:
            self.pop.sort(key=lambda x: x.fitness)
        else:
            self.pop.sort(key=lambda x: x.fitness, reverse=True)

    def breed(self):
        childdna = []
        for i in range(self.survivornum, self.popnum):
            parent1 = self.pop[random.randint(0, self.parentnum-1)]
            parent2 = self.pop[random.randint(0, self.parentnum-1)]
            dna = self.crossoverfun(parent1.dna, parent2.dna)
            dna = self.mutationfun(dna)
            childdna.append(dna)

        for i in range(self.survivornum, self.popnum):
            self.pop[i].clear()
            self.pop[i].dna = childdna[i-self.survivornum]

    def display(self, block=True):
        for i, individual in enumerate(self.pop):
            individual.display(str(i), block)


class IndividualPoly:

    def __init__(self, polynum, verticenum):
        self.polynum = polynum
        self.verticenum = verticenum
        self.fitness = False
        self.img = False
        self.dna = False

    def randomize(self):
        self.dna = np.random.random(self.verticenum*self.polynum*2 + self.polynum*4)

    def zero(self):
        self.dna = np.zeros(self.verticenum*self.polynum*2 + self.polynum*4)

    def zerocolonly(self):
        self.dna = np.random.random(self.verticenum*self.polynum*2 + self.polynum*4)
        self.dna[self.verticenum*self.polynum*2::] = 0

    def draw(self, shape, dtype=np.uint8, colorspace="RGB"):

        ptsarray = np.array(self.dna[0:self.verticenum*self.polynum*2].reshape([self.polynum, self.verticenum, 2]))
        colarray = np.array(self.dna[self.verticenum*self.polynum*2::].reshape([self.polynum, 4]))

        ptsarray[:,:,0] = ptsarray[:,:,0]*shape[0]
        ptsarray[:,:,1] = ptsarray[:,:,1]*shape[1]
        ptsarray = ptsarray.astype(int)

        colarray[:,:3] = colarray[:,:3]*255
        colarray[:,:3] = colarray[:,:3].astype(int)

        img = np.zeros([shape[1], shape[0], 3], dtype=dtype)

        for i in range(0, self.polynum):
            overlay = img.copy()
            pts = ptsarray[i]
            color = tuple(colarray[i,0:3].tolist())
            alpha = colarray[i, 3]/1.2
            cv2.fillPoly(overlay, [pts], color)
            img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

        if colorspace == "HSV":
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        elif colorspace == "LAB":
            img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

        self.img = img

    def display(self, windowname, block=True):
        cv2.imshow(windowname, self.img)
        if block:
            cv2.waitKey()
        else:
            cv2.waitKey(1)

    def clear(self):
        self.fitness = False
        self.img = False
        self.dna = False

    def save(self, filename):
        cv2.imwrite(filename, self.img)


class IndividualRectangle(IndividualPoly):
    def __init__(self, polynum):
        IndividualPoly.__init__(self, polynum, 2)

    def draw(self, shape, dtype=np.uint8, colorspace="RGB"):
        ptsarray = np.array(self.dna[0:self.verticenum * self.polynum * 2].reshape([self.polynum, self.verticenum, 2]))
        colarray = np.array(self.dna[self.verticenum * self.polynum * 2::].reshape([self.polynum, 4]))

        ptsarray[:, :, 0] = ptsarray[:, :, 0] * shape[0]
        ptsarray[:, :, 1] = ptsarray[:, :, 1] * shape[1]
        ptsarray = ptsarray.astype(int)

        colarray[:, :3] = colarray[:, :3] * 255
        colarray[:, :3] = colarray[:, :3].astype(int)

        img = np.zeros([shape[1], shape[0], 3], dtype=dtype)

        for i in range(0, self.polynum):
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
        ptsarray = np.array(self.dna[0:self.verticenum * self.polynum * 2].reshape([self.polynum, self.verticenum, 2]))
        colarray = np.array(self.dna[self.verticenum * self.polynum * 2::].reshape([self.polynum, 4]))

        ptsarray[:, :, 0] = ptsarray[:, :, 0] * shape[0]
        ptsarray[:, :, 1] = ptsarray[:, :, 1] * shape[1]
        ptsarray = ptsarray.astype(int)

        colarray[:, :3] = colarray[:, :3] * 255
        colarray[:, :3] = colarray[:, :3].astype(int)

        img = np.zeros([shape[1], shape[0], 3], dtype=dtype)

        for i in range(0, self.polynum):
            overlay = img.copy()
            pts = ptsarray[i]
            color = tuple(colarray[i, 0:3].tolist())
            alpha = colarray[i, 3] / 1.2
            cv2.circle(overlay, tuple(pts[0]), int(pts[1,0]/2), color, -1)
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        if colorspace == "HSV":
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        elif colorspace == "LAB":
            img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

        self.img = img


class ImgComparison:
    def rico_ssim(img1, img2):
        ssim = metrics.structural_similarity(img1, img2, multichannel=True)
        return ssim

    def rico_mse(image1, image2, displayFlag=False):
        diff = np.subtract(image1, image2)
        err = np.sum(np.square(diff))/np.prod(image1.shape)
        return err

    def rico_mse_hsv(image1, image2):
        # print(image1.shape)
        hsv1 = cv2.cvtColor(image1, cv2.COLOR_RGB2HSV)
        hsv2 = cv2.cvtColor(image2, cv2.COLOR_RGB2HSV)
        diff = np.subtract(hsv1, hsv2)
        err = np.sum(np.square(diff))/np.prod(image1.shape)
        return err

    def rico_mse_lab(image1, image2):
        lab1 = cv2.cvtColor(image1, cv2.COLOR_RGB2LAB)
        lab2 = cv2.cvtColor(image2, cv2.COLOR_RGB2LAB)
        diff = np.subtract(lab1, lab2)
        err = np.sum(np.square(diff))/np.prod(image1.shape)
        return err


class RicoGATools:
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

        return child

    def randomcrossover(mom, dad):
        child = np.zeros(mom.shape)
        for i in range(0, len(mom)):
            if random.random() < 0.5:
                child[i] = mom[i]
            else:
                child[i] = dad[i]

        return child

    def randmutation(child, mutationrate=0.05):
        """
        every dna item has a mutationrate chance of becoming a random value
        """
        mutations = 0
        mutant = np.array(child)
        for i in range(0, len(child)):
            if random.random() < mutationrate:
                mutant[i] = random.random()
                mutations += 1

        return mutant, mutations

    def randmutation_shift(child, mutationrate, shiftmax):
        """
        every dna item has a mutationrate chance of incrementing by a random percentage of shiftmax in either direction
        """
        mutations = 0
        mutant = np.array(child)
        for i in range(0, len(child)):
            if random.random() < mutationrate:
                mutant[i] += (random.random()-0.5)*shiftmax*2
                mutant[i] = max(mutant[i], 0)
                mutant[i] = min(mutant[i], 1)
                mutations += 1

        return mutant, mutations

    def randmutation_single(child, mutationrate, mutationamount):
        """
        child has a mutation rate chance of becoming a mutant, if it is than each piece of dna has a mutationamount
        chance of being randomized
        """
        mutations = 0
        mutant = np.array(child)
        if random.random() < mutationrate:
            for i in range(0, len(child)):
                if random.random() < mutationamount:
                    mutant[i] = random.random()
                    mutations += 1

        return mutant, mutations
