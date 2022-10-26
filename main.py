# FF:
#   FF1: maximize diversity on canvas
#   FF2: minimize size of generated code
from PIL import Image, ImageDraw, ImageChops
import opensimplex
import tracery
import random
import math
from itertools import repeat
from generative_object import GenerativeObject
from techniques import *
from copy import deepcopy
import multiprocessing as mpc

DIM = (1000,1000)

def evaluate(g):#id, dim, grammar):
    print("Evaluating {0}:{1}".format(g.id, g.grammar))
    #g.isEvaluated = True
    for technique in g.grammar.split(','):
        c = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        if technique == 'flow-field':
            flowField(g.image, 1, g.dim[1], g.dim[0], c)
        elif technique == 'stippled':
            stippledBG(g.image, c, g.dim)
        elif technique == 'pixel-sort':
            g.image = pixelSort(g.image)

    return g

if __name__ == "__main__":
    # tracery grammar
    rules = {
      'ordered_pattern': ['#techniques#'], 
      'techniques': ['#technique#', '#techniques#,#technique#'],
      'technique': ['stippled', 'flow-field', 'pixel-sort'],
    }
    grammar = tracery.Grammar(rules)
    #print(grammar.flatten("#ordered_pattern#"))



    #img = Image.new("RGBA", DIM, "grey")
    #draw = ImageDraw.Draw(img)

    #img2 = Image.new("RGBA", DIM, "grey")
    #draw2 = ImageDraw.Draw(img2)


    #fill = (0,0,0)

    opensimplex.seed(random.randint(0,100000))
    #flowField(draw, 1, DIM[1], DIM[0])
    #stippledBG(draw, fill)
    #print(rmsdiff(img, img2), rmsdiff(img, img3), rmsdiff(img2, img3))

    num_gens = 20
    pop_size = 50
    population = []



    #g = GenerativeObject(DIM)
    #techniques = grammar.flatten("#ordered_pattern#")
    #print(techniques)
    i = 0
    while len(population) < pop_size:
        idx = "{0}_{1}".format(gen,i)
        g = GenerativeObject(idx, DIM, grammar.flatten("#ordered_pattern#"))

        population.append(g)
        i += 1

    for gen in range(num_gens):
        print("Generation",gen)

        # evaluation
        unevaluated = list(filter(lambda x: not x.isEvaluated, population))
        with mpc.Pool(mpc.cpu_count()-1) as p:
            retval = p.starmap(evaluate, zip(unevaluated))
            for i in range(len(retval)):
                assert unevaluated[i].id == retval[i].id, "Error with ID match on re-joining."
                unevaluated[i].image = retval[i].image

        #for g in unevaluated:
        #    print("Evaluating {0}:{1}".format(g.id, g.grammar))
        #    evaluate(g)

        # selection
        # crossover
        # mutation


        # filling in
        i = 0
        while len(population) < pop_size:
            idx = "{0}_{1}".format(gen,i)
            g = GenerativeObject(idx, DIM, grammar.flatten("#ordered_pattern#"))
            population.append(g)
            i += 1

        print("Population:")
        for p in population:
            print(p.id, p.isEvaluated, p.grammar)
        print("---")

        # pair-wise comparison
        compared = {}
        for p in population:
            psum = 0
            for p2 in population:
                if p != p2:
                    id1 = "{0}:{1}".format(p.id, p2.id)
                    id2 = "{0}:{1}".format(p2.id, p.id)
                    keys = compared.keys()
                    if not id1 in keys or not id2 in keys:
                        diff = rmsdiff(p.image, p2.image)
                        compared[id1] = True
                        psum += diff
            psum /= (len(population)-1)
            p.setFitness(psum)
                
        population.sort(key=lambda x: x.fitness, reverse=True)
        print("Generation {0} best fitness: {1}, {2}, {3}".format(gen, population[0].fitness, population[0].grammar, population[0].id))

        # elite preservation
        if (gen < num_gens - 2):
            for j in range(pop_size-1, 0, -1):
                del population[j]

    # Final evaluation
    unevaluated = list(filter(lambda x: not x.isEvaluated, population))
    with mpc.Pool(mpc.cpu_count()-1) as p:
        retval = p.starmap(evaluate, zip(unevaluated))
        for i in range(len(retval)):
            assert unevaluated[i].id == retval[i].id, "Error with ID match on re-joining."
            unevaluated[i].image = retval[i].image

    # Print out last generation
    for i in range(len(population)):
        print(population[i].fitness, population[i].grammar)
        population[i].image.save("img-{0}.png".format(population[i].id))

    print("End of line.")

