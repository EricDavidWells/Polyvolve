# Polyvolve
A genetic algorithm that recreates images using transparent polygons.  See http://ericdavidwells.com/projects/polyvolve for examples and details

# Dependencies
* Python 3.7.2
* OpenCV 4.1.1.26
* Matplotlib 3.1.2
* Scikit-image 0.16.2

# Getting Started
To see the program in action, clone the repository, set up the dependencies, and run `python main.py -d -p`.

# Command Line Options

`-h`, `--help`\
* show this help message and exit

`--image_path IMAGE_PATH`\
* Path to source image.

`--output_dir OUTPUT_DIR`\
* Where output images should be stored.

`-x X`\
* Output image width.

`-y Y` 
* Height of output image. If not set, automatically calculates using the input image's ratio.

`--display_x DISPLAY_X`
* Width of displayed image.

`--display_y DISPLAY_Y`\
* Height of displayed image. If not set, automatically calculates using the input image's ratio.\

`--calc_x CALC_X`\
`--calc_y CALC_Y`\

`--num_individuals NUM_INDIVIDUALS`\
* Population of individuals (aka images) present at each time step.

`--num_polygons NUM_POLYGONS`\
* Number of polygons per individual

`--num_vertices NUM_VERTICES`\
* Number of vertices per polygon

`--mutation_rate MUTATION_RATE`\
* Rate at which individuals mutate.

`--mutation_amount MUTATION_AMOUNT`\
* The degree to which an individual is mutated.

`--p_survival P_SURVIVAL`\
* Percentage of individuals that survive to the next step.

`--p_reproduce P_REPRODUCE`\
Percentage of individuals which reproduce each step.

`--init_fn {random,zero,zero_color,random_rect,random_circle}`\

`--fitness_fn {ssim,mse,mseLAB}`\
* Function for determining the fitness of a given individual.

`--crossover_fn {random,two_point}`\
* Function for determining how individuals cross over (eg. what their children look like.)

`--mutation_fn {random_shift}`
* Function for mutating an individual.

`-s`                   
* If selected, will periodically save the best individual, as an image, to diskm as well as a graph of fitness over time.

`-p`\             
* If selected, will display a plot of the program fitness as the program executes.

`-d`                   
* If selected, will periodically display the best individual.\

`--minimize_fitness`    
* If selected, chooses individuals with minimum fitness.

`--max_generations MAX_GENERATIONS`\
* Maximum number of generations before the program finishes execution.

`--generations_per_display GENERATIONS_PER_DISPLAY`\
* Number of generations between each image being displayed.

`--generations_per_plot GENERATIONS_PER_PLOT`\
* Number of generations between each plot update.

`--generations_per_save GENERATIONS_PER_SAVE`\
* Number of generations between saving images.

`--run_to_fitness RUN_TO_FITNESS`\
* Run until a fitness of the given value is reached (should be from 0 to 1).

`--save_last`
* Saves final frame.

# Image Source
starrynight.jpg was downloaded from https://en.wikipedia.org/wiki/File:Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg.
