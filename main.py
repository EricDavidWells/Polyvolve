from rico_imgGA import *
import cv2
from matplotlib import pyplot as plt
import time


# helper functions created using global variables to feed into population class.  they allow for customization of the
# functions created in rico_imgGA file while ensuring that the input and outputs are what the Population class expects
def init_random():
    individual = IndividualPoly(num_polygons, num_vertices)
    individual.randomize()
    return individual


def init_zero():
    individual = IndividualPoly(num_polygons, num_vertices)
    individual.zero()
    return individual


def init_zero_color():
    individual = IndividualPoly(num_polygons, num_vertices)
    individual.zerocolonly()
    return individual


def init_random_rect():
    individual = IndividualRectangle(num_polygons)
    individual.randomize()
    return individual


def init_random_circle():
    individual = IndividualCircle(num_polygons)
    individual.randomize()
    return individual


def fitness_ssim(individual):
    if not individual.img or individual.img.shape[0:2] != calc_shape:
        individual.draw(calc_shape)
    err = ImgComparison.rico_ssim(individual.img, img_calc)
    return err


def fitness_mse(individual):
    if not individual.img or individual.img.shape[0:2] != calc_shape:
        individual.draw(calc_shape)
    err = ImgComparison.rico_mse(individual.img, img_calc)
    return err


def fitness_mseLAB(individual):
    if not individual.img or individual.img.shape[0:2] != calc_shape:
        individual.draw(calc_shape)
    err = ImgComparison.rico_mse_lab(individual.img, img_calc)
    return err


def mutation_randshift(dna):
    mutantdna, mutations = RicoGATools.randmutation_shift(dna, mutation_rate, mutation_amount)
    return mutantdna


def parse_args(init_functions, fitness_functions, crossover_functions, mutation_functions):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="./starrynight.jpg",
                        help="Path to source image.")
    parser.add_argument("--output_dir", type=str, default="./output/",
                        help="Where output images should be stored.")
    parser.add_argument("-x", type=int, default=650,
                        help="Output image width.")
    parser.add_argument("-y", type=int, default=None,
                        help="Height of output image. If not set,"
                             " automatically calculates using the input image's ratio.")
    parser.add_argument("--display_x", type=int, default=300,
                        help="Width of displayed image.")
    parser.add_argument("--display_y", type=int, default=None,
                        help="Height of displayed image. If not set,"
                             " automatically calculates using the input image's ratio.")
    parser.add_argument("--calc_x", type=int, default=75)
    parser.add_argument("--calc_y", type=int, default=None)
    parser.add_argument("--num_individuals", type=int, default=100,
                        help="Population of individuals (aka images) present at each time step.")
    parser.add_argument("--num_polygons", type=int, default=150,
                        help="Number of polygons per individual")
    parser.add_argument("--num_vertices", type=int, default=3,
                        help="Number of vertices per polygon")
    parser.add_argument("--mutation_rate", type=float, default=0.01,
                        help="Rate at which individuals mutate.")
    parser.add_argument("--mutation_amount", type=float, default=0.15,
                        help="The degree to which an individual is mutated.")
    parser.add_argument("--p_survival", type=float, default=0.15,
                        help="Percentage of individuals that survive to the next step.")
    parser.add_argument("--p_reproduce", type=float, default=0.15,
                        help="Percentage of individuals which reproduce each step.")
    parser.add_argument("--init_fn", default="init_random", choices=init_functions.keys())
    parser.add_argument("--fitness_fn", default="fitness_ssim", choices=fitness_functions.keys(),
                        help="Function for determining the fitness of a given individual.")
    parser.add_argument("--crossover_fn", default="random_crossover", choices=crossover_functions.keys(),
                        help="Function for determining how individuals cross over (eg. what their children look like.)")
    parser.add_argument("--mutation_fn", default="mutation_randshift", choices=mutation_functions.keys(),
                        help="Function for mutating an individual.")
    parser.add_argument("-s", dest="save", action="store_true",
                        help="If selected, will periodically save the best individual, as an image, to diskm as well as"
                             " a graph of fitness over time.")
    parser.add_argument("-p", dest="plot", action="store_true",
                        help="If selected, will display a plot of the program fitness as the program executes.")
    parser.add_argument("-d", dest="display", action="store_true",
                        help="If selected, will periodically display the best individual.")

    parser.add_argument("--minimize_fitness", action="store_true",
                        help="If selected, chooses individuals with minimum fitness.")
    parser.add_argument("--max_generations", type=int, default=50000,
                        help="Maximum number of generations before the program finishes execution.")
    parser.add_argument("--generations_per_display", type=int, default=1,
                        help="Number of generations between each image being displayed.")
    parser.add_argument("--generations_per_plot", type=int, default=10,
                        help="Number of generations between each plot update.")
    parser.add_argument("--generations_per_save", type=int, default=10,
                        help="Number of generations between saving images.")
    args = parser.parse_args()
    if not (args.display or args.plot or args.save):
        raise EnvironmentError("Must select one of -d, -p, or -s")

    if args.plot:
        print("Please avoid dragging the plot until you see a line - the program may crash otherwise.")

    return args


if __name__ == "__main__":
    import os

    IMAGE_SUB_DIR = "images"
    PLOT_SUB_DIR = "plots"

    init_functions = {"init_random": init_random,
                      "init_zero": init_zero,
                      "init_zero_color": init_zero_color,
                      "init_random_rect": init_random_rect,
                      "init_random_circle": init_random_circle}
    fitness_functions = {"fitness_ssim": fitness_ssim,
                         "fitness_mse": fitness_mse,
                         "fitness_mseLAB": fitness_mseLAB}
    mutation_functions = {"mutation_randshift": mutation_randshift}
    crossover_functions = {"random_crossover": RicoGATools.randomcrossover,
                           "two_point_crossover": RicoGATools.twopointcrossover}

    args = parse_args(init_functions, fitness_functions, crossover_functions, mutation_functions)

    file_name = args.image_path

    output_directory = os.path.join(args.output_dir, os.path.basename(args.image_path).split(".")[0])
    image_dir = os.path.join(output_directory, IMAGE_SUB_DIR)
    plot_dir = os.path.join(output_directory, PLOT_SUB_DIR)

    if args.save:
        os.makedirs(os.path.join(image_dir), exist_ok=True)
        os.makedirs(os.path.join(plot_dir), exist_ok=True)

    num_individuals = args.num_individuals
    p_survival = args.p_survival
    p_reproduce = args.p_reproduce

    num_polygons = args.num_polygons
    num_vertices = args.num_vertices
    init_function = init_functions[args.init_fn]

    fitness_function = fitness_functions[args.fitness_fn]

    crossover_function = crossover_functions[args.crossover_fn]

    mutation_rate = args.mutation_rate
    mutation_amount = args.mutation_amount
    mutation_function = mutation_functions[args.mutation_fn]
    eval_type = -1 if args.minimize_fitness else 1

    max_generations = args.max_generations
    generations_per_display = args.generations_per_display

    generations_per_plot = args.generations_per_plot

    num_images = 0
    max_images = max_generations // args.generations_per_save
    len_count_str = len(str(max_images))
    time_str = time.strftime("%Y%m%d_%H%M")
    generations_per_save = args.generations_per_save

    img_orig = cv2.imread(file_name)
    shape_orig = img_orig.shape
    ratio = shape_orig[0] / shape_orig[1]

    disp_shape = (args.display_x, args.display_y if args.display_y else int(args.display_x * ratio))
    calc_shape = (args.calc_x, args.calc_y if args.calc_y else int(args.calc_x * ratio))
    save_shape = (args.x, args.y if args.y else int(args.x * ratio))

    img_calc = cv2.resize(img_orig, calc_shape)
    img_calc_enlarged = cv2.resize(img_calc, disp_shape)

    if args.display:
        cv2.imshow("target", img_calc_enlarged)

    population = Population(num_individuals, init_function, fitness_function, crossover_function,
                            mutation_function, p_survival, p_reproduce, eval_type)

    plt.xlabel("Generations")
    plt.ylabel("Fitness")

    fitness_log = []
    for i in range(1, max_generations + 1):
        population.evaluate()
        population.breed()

        best = population.pop[0]
        fitness_log.append(best.fitness)

        if args.display and i % generations_per_display == 0:
            best.draw(disp_shape)
            best.display("Best", False)

        if args.plot and i % generations_per_plot == 0:
            plt.plot(fitness_log, 'b')
            plt.draw()
            plt.pause(0.001)

        if args.save and i % generations_per_save == 0:
            padded_str = str(num_images).zfill(len_count_str)

            best.draw(save_shape)
            best.save(os.path.join(image_dir, f"{time_str}_GA{padded_str}.png"))

            plt.plot(fitness_log, 'b')
            plt.savefig(os.path.join(plot_dir, f"{time_str}_PL{padded_str}.png"))
            
            if not args.display:
                fitness_str = str(fitness_log[-1] * 100)[:6].ljust(6, "0")
                percentage_str = str(i / max_images * 100)[:len_count_str+2].ljust(len_count_str + 2, "0")
                print(f"{percentage_str}% done. Current fitness is {fitness_str}%.")
            num_images += 1

        if args.display:
            print(i)

    if args.display:
        cv2.waitKey()
