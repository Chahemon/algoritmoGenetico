import numpy
import gari
import pygad
import matplotlib.pyplot

arreglo = []
while len(arreglo) < 64:
    arreglo.append([1, 0, 0])

# Arreglo con cada columna de color rojo que se va a replicar.
target_im = numpy.array(arreglo)

# Arreglo descompuesto en cromosomas, toda la información en una arreglo de una dimensión.
target_chromosome = gari.img2chromosome(target_im)

# Función para mejorar el fitness a cada generación.
def fitness_fun(solution, solution_idx):
    fitness = numpy.sum(numpy.abs(target_chromosome - solution))
    fitness = numpy.sum(target_chromosome) - fitness
    return fitness

# Función que se va a llamar a cada instancia para mostrar información y guardar imágenes del progreso.
def callback(ga_instance):
    # Mostramos la generación y el fitness.
    print("Generación = {gen}".format(gen=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

    # Si llegamos a ciertas generaciones, guardamos la imagen.
    if ga_instance.generations_completed == 100000 or ga_instance.generations_completed == 60000 or \
       ga_instance.generations_completed == 10000 or ga_instance.generations_completed == 4000 or \
       ga_instance.generations_completed == 1000 or ga_instance.generations_completed == 400 or \
       ga_instance.generations_completed == 100 or ga_instance.generations_completed == 20:
        # Adaptamos el arreglo para poder guardar la imagen.
        arregloProv = []
        for i in range(0, 64):
            for n in ga_instance.best_solution()[0]:
                arregloProv.append(n)
        # Guardamos la imagen.
        matplotlib.pyplot.imsave('solución_' + str(ga_instance.generations_completed) + '.png',
                                 gari.chromosome2img(arregloProv, [64, 64, 3]))


ga_instance = pygad.GA(num_generations=100000,
                       num_parents_mating=10,
                       fitness_func=fitness_fun,
                       sol_per_pop=20,
                       num_genes=target_im.size,
                       init_range_low=0.0,  # Rango de los valores que va a tener los numeros random
                       init_range_high=1.0,  # Rango de Arriba (Valores RGB de 0-1 de los pixeles que van a mutar).
                       mutation_percent_genes=0.01,
                       mutation_type="random",
                       mutation_by_replacement=True,
                       random_mutation_min_val=0.0,
                       random_mutation_max_val=1.0,
                       callback_generation=callback)

# Corremos el algoritmo.
ga_instance.run()

# Mostramos la gráfica con los valores de fitness con respecto al las generaciones.
ga_instance.plot_result()

# Mostramos en consola lo información de las mejores soluciones.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("El mejor valor de Fitness obtenido = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index de la mejor solución: {solution_idx}".format(solution_idx=solution_idx))

# Mostramos cual fue la mejor generación.
if ga_instance.best_solution_generation != -1:
    print("El mejor fitness fue alcanzado en la  {best_solution_generation} generación.".format(
        best_solution_generation=ga_instance.best_solution_generation))

# Guardamos un arreglo con la solución en formato de imagen (Arreglo de 3 dimensiones [64][64][3])
result = gari.chromosome2img(solution, target_im.shape)

resultResize = []
cont = 0
while cont < 64:
    resultResize.append(result)
    cont += 1

# Mostramos el resultado de la imagen.
matplotlib.pyplot.imshow(resultResize)
matplotlib.pyplot.title("Algoritmo Genético")
matplotlib.pyplot.show()