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
    parser.add_argument("--image_path", type=str, default="./starrynight.jpg")
    parser.add_argument("--output_dir", type=str, default="./output/")
    parser.add_argument("-x", type=int, default=650)
    parser.add_argument("-y", type=int, default=None)
    parser.add_argument("--display_x", type=int, default=300)
    parser.add_argument("--display_y", type=int, default=None)
    parser.add_argument("--calc_x", type=int, default=75)
    parser.add_argument("--calc_y", type=int, default=None)
    parser.add_argument("--num_individuals", type=int, default=100)
    parser.add_argument("--num_polygons", type=int, default=150)
    parser.add_argument("--num_vertices", type=int, default=3)
    parser.add_argument("--mutation_rate", type=float, default=0.01)
    parser.add_argument("--mutation_amount", type=float, default=0.15)
    parser.add_argument("--p_survival", type=float, default=0.15)
    parser.add_argument("--p_reproduce", type=float, default=0.15)
    parser.add_argument("--init_fn", default="init_random", choices=init_functions.keys())
    parser.add_argument("--fitness_fn", default="fitness_ssim", choices=fitness_functions.keys())
    parser.add_argument("--crossover_fn", default="random_crossover", choices=["random_crossover"])
    parser.add_argument("--mutation_fn", default="mutation_randshift", choices=["mutation_randshift"])
    parser.add_argument("-s", dest="save", action="store_true",
                        help="Saves image to disk.")
    parser.add_argument("-p", dest="plot", action="store_true",
                        help="Displays plot.")
    parser.add_argument("-d", dest="display", action="store_true",
                        help="Displays image.")

    parser.add_argument("--minimize_fitness", action="store_true")
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


def handle_close(event):
    exit(-1)


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

    plot_dir = os.path.join(output_directory, PLOT_SUB_DIR)
    os.makedirs(os.path.join(plot_dir), exist_ok=True)

    im_width = args.x
    image_dir = os.path.join(output_directory, IMAGE_SUB_DIR)
    os.makedirs(os.path.join(image_dir), exist_ok=True)

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
    eval_type = 1  # 1 or maximizing fitness, -1 for minimizing fitness

    max_generations = args.max_generations
    generations_per_display = args.generations_per_display
    generations_per_plot = args.generations_per_plot
    generations_per_save = args.generations_per_save
    num_images = 0
    len_count_str = len(str(max_generations // args.generations_per_save))
    fitness_log = []

    timestr = time.strftime("%Y%m%d_%H%M")
    img_orig = cv2.imread(file_name)
    shape_orig = img_orig.shape
    ratio = shape_orig[0] / shape_orig[1]
    disp_shape = (args.display_x, args.display_y if args.display_y else int(args.display_x * ratio))
    calc_shape = (args.calc_x, args.calc_y if args.calc_y else int(args.calc_x * ratio))
    save_shape = (args.x, args.y if args.y else int(args.x * ratio))

    img_calc = cv2.resize(img_orig, calc_shape)
    img_calc_enlarged = cv2.resize(img_calc, disp_shape)

    cv2.imshow("target", img_calc_enlarged)

    population = Population(num_individuals, init_function, fitness_function, crossover_function,
                            mutation_function, p_survival, p_reproduce, eval_type)

    plt.xlabel("Generations")
    plt.ylabel("Fitness")

    for i in range(0, max_generations):
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
            best.draw(save_shape)
            padded_str = str(num_images).zfill(len_count_str)
            best.save(f"{image_dir}_GA{padded_str}.png")
            plt.savefig(f"{plot_dir}_PL{padded_str}.png")
            num_images += 1

        print(i)
    cv2.waitKey()
