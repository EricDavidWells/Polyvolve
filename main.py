from rico_imgGA import *
import cv2
from matplotlib import pyplot as plt
import time


# helper functions created using global variables to feed into population class.  they allow for customization of the
# functions created in rico_imgGA file while ensuring that the input and outputs are what the Population class expects
def initrandom():
    individual = IndividualPoly(polynum, verticenum)
    individual.randomize()
    return individual


def initzero():
    individual = IndividualPoly(polynum, verticenum)
    individual.zero()
    return individual


def initzerocolor():
    individual = IndividualPoly(polynum, verticenum)
    individual.zerocolonly()
    return individual


def initrandomrec():
    individual = IndividualRectangle(polynum)
    individual.randomize()
    return individual


def initrandomcircle():
    individual = IndividualCircle(polynum)
    individual.randomize()
    return individual


def fitness_ssim(individual):
    if not individual.img or individual.img.shape[0:2] != calcshape:
        individual.draw(calcshape)
    err = ImgComparison.rico_ssim(individual.img, img_calc)
    return err


def fitness_mse(individual):
    if not individual.img or individual.img.shape[0:2] != calcshape:
        individual.draw(calcshape)
    err = ImgComparison.rico_mse(individual.img, img_calc)
    return err


def fitness_mseLAB(individual):
    if not individual.img or individual.img.shape[0:2] != calcshape:
        individual.draw(calcshape)
    err = ImgComparison.rico_mse_lab(individual.img, img_calc)
    return err


def mutation_randshift(dna):
    mutantdna, mutations = RicoGATools.randmutation_shift(dna, mutationrate, mutationamount)
    return mutantdna


if __name__ == "__main__":
    filename = "starrynight.png"
    # directory to save files in MUST HAVE two folders in it, one titled "images" and one titled "plots" if you wish
    # to save the results periodically
    savefiledirectory = r"C:\Users\Rico\Pictures\polyvolve\\" + filename.replace(".png", "")
    saveflag = False
    plotflag = False
    displayflag = True
    savesize = 650  # x-dimension resolution to save polygon images (y-dimension scaled)
    dispsize = 300  # x-dimension resolution for display during runtime (y-dimension scaled)
    calcsize = 75   # x-dimension resolution for actual calculation (y-dimension scaled)

    populationnum = 100 # number of individuals per generation
    polynum = 150   # number of polygons per individual
    verticenum = 3  # number of vertices per polygon
    mutationrate = 0.01     # dependent on mutation function
    mutationamount = 0.15   # dependent on mutation function
    survivalamt = 0.15  # percentage of individuals who survive to next generation
    parentamt = 0.15    # percentage of individuals used to create next generation
    initfunction = initrandom   # function used to initiate each individual, must have no parameters and return an individual
    # initfunction = initrandomrec
    # initfunction = initrandomcircle
    fitnessfunction = fitness_ssim  # function used to evaluate fitness of each individual, must take in individual parameter
                                    # and return a fitness value
    crossoverfunction = RicoGATools.randomcrossover # function used to create dna for new generation, must take in two
                                                    # lists of DNA and return a new list of DNA
    # crossoverfunction = RicoGATools.twopointcrossover
    mutationfunction = mutation_randshift   # function used to mutate dna, must take in a list of DNA and return a list of DNA
    evaltype = 1    # 1 or maximizing fitness, -1 for minimizing fitness

    maxgenerations = 50000  # max generations to stop algorithm
    generationsperdisp = 1  # generations per update of display image
    generationsperplot = 10 # generations per update of display plot
    generationspersave = 10 # generations per save of display image
    imgsavecount = 0
    fitlog = []

    timestr = time.strftime("%Y%m%d_%H%M")
    img_orig = cv2.imread(r"images\\" + filename)
    shape_orig = img_orig.shape
    dispshape = (dispsize, int(dispsize * shape_orig[0] / shape_orig[1]))
    calcshape = (calcsize, int(calcsize * shape_orig[0] / shape_orig[1]))
    saveshape = (savesize, int(savesize * shape_orig[0] / shape_orig[1]))

    img_calc = cv2.resize(img_orig, calcshape)
    img_calcenlarged = cv2.resize(img_calc, tuple([dispshape[0], dispshape[1]]))

    cv2.imshow("target", img_calcenlarged)

    population = Population(populationnum, initfunction, fitnessfunction, crossoverfunction,
                            mutationfunction, survivalamt, parentamt, evaltype)

    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    for i in range(0, maxgenerations):
        population.evaluate()
        population.breed()

        best = population.pop[0]
        fitlog.append(best.fitness)

        if i % generationsperdisp == 0:
            if displayflag:
                best.draw(dispshape)
                best.display("best", False)

        if i % generationsperplot == 0:
            plt.plot(fitlog, 'b')
            if plotflag:
                plt.draw()
                plt.pause(0.001)

        if i % generationspersave == 0:
            if saveflag:
                best.draw(saveshape)
                filepath = savefiledirectory + "\\images\\" + timestr + "_GA" + str(imgsavecount)
                best.save(filepath + ".png")
                imgsavecount += 1
                filepath = savefiledirectory + "\\plots\\" + timestr + "_PL" + str(imgsavecount)
                plt.savefig(filepath + ".png")

        print(i)
    cv2.waitKey()