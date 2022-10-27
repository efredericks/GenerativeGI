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

    num_gens = 25 
    pop_size = 50 
    # num_gens = 3
    # pop_size = 10
    xover_rate = 0.6
    mut_rate = 0.3
    population = []



    #g = GenerativeObject(DIM)
    #techniques = grammar.flatten("#ordered_pattern#")
    #print(techniques)

    ##### GENERATION 0
    ##### THIS ALL NEEDS TO BE REFACTORED TO BE FXN CALLS
    print("Generation",0)
    i = 0
    while len(population) < pop_size:
        idx = "{0}_{1}".format(str(0),i)
        g = GenerativeObject(idx, DIM, grammar.flatten("#ordered_pattern#"))
        population.append(g)
        i += 1

    # evaluation
    unevaluated = list(filter(lambda x: not x.isEvaluated, population))
    with mpc.Pool(mpc.cpu_count()-1) as p:
    # with mpc.Pool(4) as p:
        retval = p.starmap(evaluate, zip(unevaluated))
        for i in range(len(retval)):
            assert unevaluated[i].id == retval[i].id, "Error with ID match on re-joining."
            unevaluated[i].isEvaluated = True
            unevaluated[i].image = retval[i].image

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
    print("Generation {0} best fitness: {1}, {2}, {3}".format(0, population[0].fitness, population[0].grammar, population[0].id))
    print("---")
    #####################

    for gen in range(1,num_gens):
        print("Generation",gen)

        #for g in unevaluated:
        #    print("Evaluating {0}:{1}".format(g.id, g.grammar))
        #    evaluate(g)


        #print("Population:")
        #for p in population:
        #    print(p.id, p.isEvaluated, p.grammar)
        #print("---")


        # selection - tbd
        num_xover = int(pop_size * xover_rate)
        num_mut = int(pop_size * mut_rate)
        next_pop = []


        next_pop.append(deepcopy(population[0])) # elite

        # crossover
        for j in range(int(num_xover/2)):
            id1 = random.randint(0,len(population)-1)
            id2 = random.randint(0,len(population)-1)
            while id1 == id2:
                id2 = random.randint(0,len(population)-1)
            
            # children
            c1 = deepcopy(population[id1])
            c2 = deepcopy(population[id2])

            c1.isEvaluated = False
            c2.isEvaluated = False
            c1.id += "_c_{0}1_g{1}".format(j,gen)
            c2.id += "_c_{0}2_g{1}".format(j,gen)

            split_grammar1 = population[id1].grammar.split(",")
            split_grammar2 = population[id2].grammar.split(",")

            if len(split_grammar1) > 1 and len(split_grammar2) > 1: 
                # crossover for variable length
                # pick an index each and flop
                xover_idx1 = random.randint(1,len(split_grammar2)-1)
                xover_idx2 = random.randint(1,len(split_grammar2)-1)

                new_grammar1 = []
                new_grammar2 = []

                print(len(split_grammar1),len(split_grammar2),xover_idx1,xover_idx2)
                # up to indices
                for i in range(xover_idx1):
                    new_grammar1.append(split_grammar1[i])
                for i in range(xover_idx2):
                    new_grammar2.append(split_grammar2[i])

                # past indices
                for i in range(xover_idx1, len(split_grammar1)):
                    new_grammar2.append(split_grammar1[i])
                for i in range(xover_idx2, len(split_grammar2)):
                    new_grammar1.append(split_grammar2[i])
            
            else: # one of the genomes was length 1
                new_grammar1 = []
                new_grammar2 = []

                if len(split_grammar1) == 1:
                    new_grammar2 = split_grammar2.copy()
                    new_grammar2.insert(random.randint(0,len(split_grammar2)),split_grammar1[0])

                    new_grammar1 = split_grammar2.copy()
                    new_grammar1.insert(random.randint(0,len(split_grammar2)),split_grammar1[0])
                else:
                    new_grammar2 = split_grammar1.copy()
                    new_grammar2.insert(random.randint(0,len(split_grammar1)),split_grammar2[0])

                    new_grammar1 = split_grammar1.copy()
                    new_grammar1.insert(random.randint(0,len(split_grammar1)),split_grammar2[0])


            c1.grammar = ",".join(new_grammar1)
            c2.grammar = ",".join(new_grammar2)
            next_pop.append(c1)
            next_pop.append(c2)

            print("---")
            print(c1.id, c1.grammar, population[id1].id, population[id1].grammar)
            print(c2.id, c2.grammar, population[id2].id, population[id2].grammar)
            print("---")


        # mutation
        for j in range(num_mut):
            pop_id = random.randint(0,len(population)-1)
            mutator = deepcopy(population[pop_id])
            mutator.id += "_m_{0}_g{1}".format(j,gen)
            #mutator.image = Image.new("RGBA", DIM, "black")
            # leaving the 'old' image makes it look neater imo...
            mutator.isEvaluated = False

            split_grammar = mutator.grammar.split(",")
            mut_idx = random.randint(0,len(split_grammar)-1)
            split_grammar[mut_idx] = rules['technique'][random.randint(0,len(rules['technique'])-1)]
            mutator.grammar = ",".join(split_grammar)

            # print(mutator.id, mutator.grammar, population[pop_id].id, population[pop_id].grammar)

            next_pop.append(mutator)

        # filling in
        i = 0
        while len(next_pop) < pop_size:
            idx = "{0}_{1}".format(gen,i)
            g = GenerativeObject(idx, DIM, grammar.flatten("#ordered_pattern#"))
            next_pop.append(g)
            i += 1

        # evaluation
        unevaluated = list(filter(lambda x: not x.isEvaluated, next_pop))
        with mpc.Pool(mpc.cpu_count()-1) as p:
            retval = p.starmap(evaluate, zip(unevaluated))
            for i in range(len(retval)):
                assert unevaluated[i].id == retval[i].id, "Error with ID match on re-joining."
                unevaluated[i].isEvaluated = True
                unevaluated[i].image = retval[i].image

        # pair-wise comparison
        compared = {}
        for p in next_pop:
            psum = 0
            for p2 in next_pop:
                if p != p2:
                    id1 = "{0}:{1}".format(p.id, p2.id)
                    id2 = "{0}:{1}".format(p2.id, p.id)
                    keys = compared.keys()
                    if not id1 in keys or not id2 in keys:
                        diff = rmsdiff(p.image, p2.image)
                        compared[id1] = True
                        psum += diff
            psum /= (len(next_pop)-1)
            p.setFitness(psum)
                
        next_pop.sort(key=lambda x: x.fitness, reverse=True)
        print("Generation {0} best fitness: {1}, {2}, {3}".format(gen, population[0].fitness, population[0].grammar, population[0].id))
        print("---")

    
        del population
        population = next_pop
        #for j in range(pop_size-1, -1, -1):
        #    del population[j]
        # elite preservation
        #if (gen < num_gens - 1):
        #    for j in range(pop_size-1, 0, -1):
        #        del population[j]

    # Final evaluation
    unevaluated = list(filter(lambda x: not x.isEvaluated, population))
    with mpc.Pool(mpc.cpu_count()-1) as p:
        retval = p.starmap(evaluate, zip(unevaluated))
        for i in range(len(retval)):
            assert unevaluated[i].id == retval[i].id, "Error with ID match on re-joining."
            unevaluated[i].image = retval[i].image

    # Print out last generation
    print("Final output:")
    for i in range(len(population)):
        print(population[i].id, population[i].fitness, population[i].grammar)
        population[i].image.save("img-{0}.png".format(population[i].id))
    print("---")

    print("End of line.")

