from rico_imgGA import *
import cv2
from matplotlib import pyplot as plt
import time


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
    timestr = time.strftime("%Y%m%d_%H%M")
    savefiledirectory = r"C:\Users\Rico\Pictures\polyvolve\\" + filename.replace(".png", "")
    saveflag = True
    plotflag = False
    displayflag = True
    savesize = 650
    dispsize = 300
    calcsize = 75

    populationnum = 100
    polynum = 150
    verticenum = 10
    mutationrate = 0.01
    mutationamount = 0.15
    survivalamt = 0.15
    parentamt = 0.15
    # initfunction = initrandom
    # initfunction = initrandomrec
    initfunction = initrandomcircle
    fitnessfunction = fitness_ssim
    # crossoverfunction = RicoGATools.randomcrossover
    crossoverfunction = RicoGATools.twopointcrossover
    mutationfunction = mutation_randshift
    evaltype = 1

    maxgenerations = 50000
    generationsperdisp = 1
    generationsperplot = 10
    generationspersave = 10
    imgsavecount = 0
    fitlog = []

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