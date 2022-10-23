# FF:
#   FF1: maximize diversity on canvas
#   FF2: minimize size of generated code
from PIL import Image, ImageDraw, ImageChops
import opensimplex
import tracery
import random
import math
from generative_object import GenerativeObject
from techniques import rmsdiff
from copy import deepcopy

DIM = (1000,1000)

if __name__ == "__main__":
    # tracery grammar
    rules = {
      'origin': '#hello# #location#',
      'hello': ['hello', 'greetings', 'howdy', 'hey'],
      'location': ['earth', 'world', 'there'],

      'ordered_pattern': ['#techniques#'], 
      'techniques': ['#technique#', '#techniques#,#technique#'],
      'technique': ['stippled', 'flow-field'],#, '#technique#'],
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
    population = []

    #g = GenerativeObject(DIM)
    #techniques = grammar.flatten("#ordered_pattern#")
    #print(techniques)

    for gen in range(num_gens):
        print("Generation",gen)

        #for i in range(0,pop_size):
        i = 0
        while len(population) < pop_size:
            idx = "{0}_{1}".format(gen,i)
            g = GenerativeObject(idx, DIM, grammar.flatten("#ordered_pattern#"))
            if not g.isEvaluated:
                print("Evaluating {0}:{1}".format(idx, g.grammar))
                g.evaluate()

            population.append(g)
            i += 1

        print("Population:")
        for p in population:
            print(p.id, p.isEvaluated, p.grammar)
        print("---")

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
        print("Generation {0} best fitness: {1}, {2}".format(gen, population[0].fitness, population[0].grammar))

        if (gen < num_gens - 2):
            for j in range(pop_size-1, 0, -1):
                del population[j]
          #population = population[1:]
        #population[0].image.show()

    # Final evaluation
    for i in range(len(population)):
        print(population[i].fitness, population[i].grammar)
        population[i].image.save("img-{0}.png".format(population[i].id))

    print("Done")

