"""
    Functions associated with the evolutionary process are stored here.
"""

import copy
import math
import random

import os

from PIL import Image
import tracery
from techniques import *
import cv2
from scipy.spatial import distance as dist

from settings import *

from meanDiffModel.useMeanToClassifyTestTensorSingleImageFinal import Net

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

args = ""
rng = None
lexicase_ordering = []
glob_fit_indicies = []

glob_cur_gen = 0

##########################################################################################


class ExperimentSettings(object):
    num_environments = 1
    num_objectives = num_environments * 7  # 4 ways to measure fitness per environment.

    args = ""
    rng = None

    treatments = [
        "baseline",  #0
    ]
    num_objectives = 6

    rules = rules
    grammar = tracery.Grammar(rules)

    DIM = DIM 


##########################################################################################
# Logging Methods
def cleanFitnessFile(filepath, trtmnt=-1, rep=-1, last_gen=-1):
    """ Clean up the fitness file to remove generations that we do not 
        have the population for as we only log every X generations.
    
    Args:
        filepath: what is the base folder to look for the file
        trtment: what treatment number
        rep: what replicate number
        last_gen: how far did we get with logging the population
    
    """
    data = []
    with open(filepath + "/{}_{}_fitnesses.dat".format(trtmnt, rep), "r") as f:
        # Read in the header row.
        data.append(f.readline())

        for line in f.readlines():
            spl_line = line.split(',')

            # Determine if we should stop processing
            # data as we've exceeded the last
            # generation we have data for.
            if int(spl_line) > last_gen:
                break

            data.append(line)

    with open(filepath + "/{}_{}_fitnesses.dat".format(trtmnt, rep), "w") as f:
        for d in data:
            f.write(d)


class Logging(object):
    """ Handle logging of information for evolution. """

    lexicase_information = []

    @classmethod
    def writeLexicaseOrdering(cls, filename):
        """ Write out the ordering of the fitness metrics selected per generation with lexicase. """
        if not os.path.exists(filename):
            # File does not yet exist, write headers.
            with open(filename, "w") as f:
                # Write Headers
                f.write("Gen, Sel_Event, Objective, Individuals, Sel_Ind\n")

        # Write out the information.
        with open(filename, "a") as f:
            for line in cls.lexicase_information:
                f.write(','.join(str(i) for i in line) + "\n")

        # Clear the lexicase information since we wrote it to the file.
        cls.lexicase_information = []

    @classmethod
    def writePopulationInformation(cls, filename, population):
        with open(filename, "w") as f:
            for p in population:
                f.write(f"{p._id} \t {p.fitness.values} \t {p.grammar}\n")


def writeHeaders(filename, num_objectives):
    """ Write out the headers for a logging file. """
    # pass
    with open(filename, "w") as f:
        f.write("Gen,Ind,ID")
        for i in range(num_objectives):
            f.write(",Fit_{}".format(i))
        f.write("\n")


def writeGeneration(filename, generation, individuals):
    """ Write out the fitness information for a generation. """
    # pass
    with open(filename, "a") as f:
        for i, ind in enumerate(individuals):
            f.write(str(generation) + "," + str(i) + "," + str(ind._id) + ",")
            f.write(",".join(str(f) for f in ind.fitness.values))
            f.write("\n")


##########################################################################################


# Non-class methods specific to the problem at hand.
def initIndividual(ind_class):
    return ind_class(ExperimentSettings.DIM,
                     ExperimentSettings.rng,
                     ExperimentSettings.grammar.flatten("#ordered_pattern#"))


def evaluate_individual(g):
    """ Wrapper to evaluate an individual.  

    Args:
        individual: arguments to pass to the simulation

    Returns:
        image an individual generates
    """
    for technique in g.grammar.split(','):
        _technique = technique.split(":")  # split off parameters
        c = (g.rng.randint(0,
                            255), g.rng.randint(0,
                                                 255), g.rng.randint(0, 255))
        if _technique[0] == 'flow-field':
            flowField(g.image, g.rng, 1, g.dim[1], g.dim[0], c, _technique[1],
                      _technique[2], _technique[2])
        elif _technique[0] == 'stippled':
            stippledBG(g.image, g.rng, c, g.dim)
        elif _technique[0] == 'pixel-sort':
            # 1: angle
            # 2: interval
            # 3: sorting algorithm
            # 4: randomness
            # 5: character length
            # 6: lower threshold
            # 7: upper threshold
            g.image = pixelSort(g.image, g.rng, _technique[1:])

        elif _technique[0] == 'dither':
            if _technique[1] == 'grayscale':
                g.image = convert_grayscale(g.image, rng)
            elif _technique[1] == 'halftone':
                g.image = convert_halftoning(g.image, rng)
            elif _technique[1] == 'dither':
                g.image = convert_dithering(g.image, rng)
            elif _technique[1] == 'primaryColors':
                g.image = convert_primary(g.image, rng)
            else:
                g.image = simpleDither(g.image, rng)
        elif _technique[0] == 'wolfram-ca':
            WolframCA(g.image, rng, _technique[1])
        elif _technique[0] == 'drunkardsWalk':
            drunkardsWalk(g.image, rng, palette=_technique[1])
        elif _technique[0] == 'flow-field-2':
            flowField2(g.image, rng, _technique[1], _technique[2], _technique[3],
                       _technique[4])
        elif _technique[0] == 'circle-packing':
            circlePacking(g.image, rng, _technique[1], _technique[2])
        elif _technique[0] == 'rgb-shift':
            g.image = RGBShift(g.image, rng, float(_technique[1]), float(_technique[2]), float(_technique[3]), float(_technique[4]), float(_technique[5]), float(_technique[6]), float(_technique[7]), float(_technique[8]), float(_technique[9]))
        elif _technique[0] == 'noise-map':
            g.image = noiseMap(g.image, rng, _technique[1], float(_technique[2]), float(_technique[3]), float(_technique[4]))
        elif _technique[0] == 'oil-painting-filter':
            g.image = openCV_oilpainting(g.image, rng, int(_technique[1]))
        elif _technique[0] == 'watercolor-filter':
            g.image = openCV_watercolor(g.image, rng, int(_technique[1]), float(_technique[2]))
        elif _technique[0] == 'pencil-filter':
            g.image = openCV_pencilSketch(g.image, rng, int(_technique[1]), float(_technique[2]), float(_technique[3]), _technique[4])
        elif _technique[0] == 'walkers':
            walkers(g.image, rng, palette=_technique[1], num_walkers=int(_technique[2]), walk_type=_technique[3])
        elif _technique[0] == 'basic_trig':
            basic_trig(g.image, rng, palette=_technique[1], num_to_draw=int(_technique[2]), drawtype=_technique[3])
        
    return g


# Compare each population member to each other population member (this one uses RMS difference)
# and set its fitness to be the greatest 'difference'
def pairwiseComparison(_population):
    maxUniques = 0
    maxDiff = 0.0
    compared = {}
    for p in _population:
        maxUniques = len(set(p.grammar.split(',')))
        psum = 0

        # image is the background color with no changes - weed out
        numblack = count_nonblack_pil(p.image)
        if numblack == 0:
            p.setFitness(0.0)
        else:
            for p2 in _population:
                if p != p2:
                    maxUniques2 = len(set(p2.grammar.split(',')))
                    if maxUniques2 > maxUniques:
                        maxUniques = maxUniques2

                    id1 = "{0}:{1}".format(p._id, p2._id)
                    id2 = "{0}:{1}".format(p2._id, p._id)
                    keys = compared.keys()
                    if not id1 in keys or not id2 in keys:
                        diff = rmsdiff(p.image, p2.image)

                        if (diff > maxDiff):
                            maxDiff = diff
                        compared[id1] = True
                        psum += diff
            psum /= (len(_population) - 1)
            p.setFitness(psum)

    # actual fitness?
    fitnesses = []
    for p in _population:
        try:
            fitness = p.getFitness() / maxDiff
        except ZeroDivisionError:
            print(p._id, maxDiff)#, maxUniques)
            fitness = 0.0
        p.setFitness(fitness)
        fitnesses.append(fitness)

    return fitnesses

def chebyshev(_population):
    maxDiff = 0.0
    compared = {}
    for p in _population:
        psum = 0

        # image is the background color with no changes - weed out
        numblack = count_nonblack_pil(p.image)
        if numblack == 0:
            p.setFitness(0.0)
        else:
            for p2 in _population:
                if p != p2:
                    id1 = "{0}:{1}".format(p._id, p2._id)
                    id2 = "{0}:{1}".format(p2._id, p._id)
                    keys = compared.keys()
                    if not id1 in keys or not id2 in keys:
                        hist1 = cv2.calcHist(np.asarray(p.image), [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
                        hist1 = cv2.normalize(hist1, hist1).flatten()
                        hist2 = cv2.calcHist(np.asarray(p2.image), [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
                        hist2 = cv2.normalize(hist2, hist2).flatten()

                        diff = dist.chebyshev(hist1, hist2)
                        # diff = rmsdiff(p.image, p2.image)

                        if (diff > maxDiff):
                            maxDiff = diff
                        compared[id1] = True
                        psum += diff
            psum /= (len(_population) - 1)
            p.setFitness(psum)

    # aggregate fitness
    fitnesses = []
    for p in _population:
        try:
            fitness = p.getFitness() / maxDiff
        except ZeroDivisionError:
            print(p._id, maxDiff)#, maxUniques)
            fitness = 0.0
        p.setFitness(fitness)
        fitnesses.append(fitness)

    return fitnesses


# Compare each population member's genome to see how many unique genes it has.
# This is a minimization objective as we want to select individuals with a low score since
# we'll keep a count of how many others have the same gene.  It's a histogram for each
# unique copy of a gene.
def uniqueGeneCount(_population):
    genes = {}
    for p in _population:
        ind_genes = p.grammar.split(',')

        # Add the occurences of each gene to the genes dictionary.
        for ig in ind_genes:
            if ig not in genes:
                genes[ig] = 1
            else:
                genes[ig] += 1

    # Tally each individuals scores based on the sweep of genes
    # in the population.
    fitnesses = []
    for p in _population:
        ind_genes = p.grammar.split(',')

        fitnesses.append(0)
        for ig in ind_genes:
            fitnesses[-1] += genes[ig]

    return fitnesses


# Find how many unique techniques and individual has.
def numUniqueTechniques(_population):
    fitnesses = []
    for p in _population:
        techniques = []
        for technique in p.grammar.split(','):
            techniques.append(technique.split(":")[0])
        # print(techniques)
        fitnesses.append(len(set(techniques)))
    return fitnesses

# Source: https://stackoverflow.com/a/52879133
def score_triadic_color_alignment(_population):
  """ Score each individual based on their triadic color palette.  Find the dominant color in an image 
      and then identify the three complimentary colors.  Returned score will be how closely the three 
      primary colors in the image align with a triadic color palette.  
  """
  fitnesses = []
  
  for p in _population:
      
    # Resize the image if we want to save time.
    #Resizing parameters
    width, height = 150, 150
    image = p.image.copy()
    image.thumbnail((width, height), resample=0)
    
    # Good explanation of how HSV works to find complimentary colors.
    # https://stackoverflow.com/a/69880467
    image.convert('HSV')
    
    #image = image.resize((width, height), resample = 0)
    #Get colors from image object
    pixels = image.getcolors(width * height)
    #Sort them by count number(first element of tuple)
    sorted_pixels = sorted(pixels, key=lambda t: t[0])
    
    # Get the most frequent colors
    # Filter out black if it is the dominant color since our background is black.
    top_colors = sorted_pixels[-4:]
    if top_colors[-1][1] == (0,0,0,255):
        top_colors = top_colors[:-1]
    else:
        top_colors = top_colors[-2:]
        
    # Sort the colors by hue (ascending).
    top_colors = sorted(top_colors, key=lambda x: x[1][0])
    
    # Assess how closely the three colors hue align with a 60 degree separation.
    # This would be a difference of 85 in the HSV hue value between each color.
    if len(top_colors) > 2:
        avg_distance = sum([math.fabs(top_colors[i][1][0] - top_colors[(i+1)%3][1][0]) if i != 2 else math.fabs(255-top_colors[i][1][0] + top_colors[(i+1)%3][1][0]) for i in range(3)])/3
    else:
        avg_distance = 255
    
    fitnesses.append(math.fabs(255/3 - avg_distance))

  return fitnesses

# Source: https://stackoverflow.com/a/52879133
def hsv_color_list(image):
  """ Get the list of colors present in an image object by HSV value.  
  """
  # Resize the image if we want to save time.
  #Resizing parameters
  width, height = image.width, image.height
  image = image.copy()
  image.thumbnail((width, height), resample=0)
  
  # Good explanation of how HSV works to find complimentary colors.
  # https://stackoverflow.com/a/69880467
  image.convert('HSV')
  
  #image = image.resize((width, height), resample = 0)
  #Get colors from image object
  pixels = image.getcolors(width * height)
  #Sort them by count number(first element of tuple)
  sorted_pixels = sorted(pixels, key=lambda t: t[0])
  
  for color in sorted_pixels:
    print(color)
    
  # Print the total number of pixels.
  print(sum([color[0] for color in sorted_pixels]))
  return sorted_pixels

def score_art_tf(_population):
    #Load the saved means
    tensor_folder = './meanDiffModelV2/'
    load_path = os.path.join(tensor_folder, 'mean_features_by_label.pt')

    fitnesses = []

    # Load the dictionary containing the mean features
    mean_features_by_label = torch.load(load_path)
    #print(f"Mean features by label have been loaded from {load_path}")

    net = Net()

    # modelLocation = './meanDiffModel/paintingVsSculpture.pth'
    modelLocation = './meanDiffModelV2/artVsRandomNoise.pth'
   
    net.load_state_dict(torch.load(modelLocation, map_location=torch.device('cpu')))

    transform = transforms.Compose(
        [transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    for p in _population:
        # Load specific image
        #image_path = "./img/img-{0}.jpg".format(i)
        #image_path = "img/img-86.png" #"./notArtImage.jpg"
        #image = Image.open(image_path)

        # handle alpha issue 
        image = p.image
        if image.mode == 'RGBA':
            # Drop the alpha channel
            image = image.convert('RGB')

        image = transform(image)    # Apply the transformation
        image = image.unsqueeze(0)  # Add batch dimension

        # Extract means directly by keys if you know them
        artMean = mean_features_by_label['art']
        notArtMean = mean_features_by_label['notArt']

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            outputs, tensor_before_fc = net(image)
            artDifference = torch.sum(torch.abs(tensor_before_fc - artMean))
            notArtDifference = torch.sum(torch.abs(tensor_before_fc - notArtMean))
            fitnesses.append(artDifference.item())

            #print("ART MEANS", p._id, artDifference, notArtDifference, artMean, notArtMean)

            #if(artDifference < notArtDifference):
            #    #predict art
            #    myPredicted = 0
            #else:
            #    #predict not art
            #    myPredicted = 1

            # For classification using the CNN, the class with the highest energy is the class chosen for prediction
            #_, predicted = torch.max(outputs.data, 1)

        #If this outputs 0 if the NN thinks its art, 1 if not art 
        #print('0 is art, 1 is not art')
        #print(f'Predicted = {predicted}')

        #fitnesses.append(myPredicted)

        #if(artDifference < notArtDifference):
        #     print('Closer to the art mean')
        #else:
        #     print('Closer to the not art mean')

        #tensor_file_path = f"./tensorGenerated.pt"  # replace with your desired file path
        #torch.save(tensor_before_fc, tensor_file_path)
        #print(f"Tensor saved to {tensor_file_path}")
        #print("---")

    return fitnesses
  
def score_negative_space(_population, target_percent=.7, primary_black=True):
  """ Score each individual based on their negative space compared to a target percentage.  
    Assess how closely the provided image matches the target percentage for negative space. 
  
  Args:
    _population: list of individuals to score
    target_percent: target percentage of negative space in the image
    primary_black: boolean indicating whether the primary color should be black or the top color in the image.
  """
  fitnesses = []
  
  for p in _population:
    color_distribution = hsv_color_list(p.image)
  
    total_pixels = sum([color[0] for color in color_distribution])
  
    negative_space_pixels = 0
    if primary_black:
        # The primary color is black, so the negative space is whatever the distribution of black is.
        for color in color_distribution:
            if color[1] == (0,0,0,255):
                negative_space_pixels = color[0]
                break
            else:
                # The primary color is not black, so the negative space is whatever the distribution of the top color is.
                negative_space_pixels = color[0]
    
    negative_space_percent = negative_space_pixels / total_pixels
  
    fitnesses.append(math.fabs(target_percent - negative_space_percent))

  return fitnesses

# Perform single-point crossover
def singlePointCrossover(ind1, ind2):
    # children
    c1 = copy.deepcopy(ind1)

    # clear the canvas if the command line parameter is set
    if args.clear_canvas:
        c1.image = Image.new("RGBA", DIM, "black")
        c1.isEvaluated = False

    split_grammar1 = ind1.grammar.split(",")
    split_grammar2 = ind2.grammar.split(",")

    if len(split_grammar1) > 1 and len(split_grammar2) > 1:
        # crossover for variable length
        # pick an index each and flop
        xover_idx1 = ind1.rng.randint(1, len(split_grammar1) - 1)
        xover_idx2 = ind2.rng.randint(1, len(split_grammar2) - 1)

        new_grammar1 = []
        # up to indices
        for i in range(xover_idx1):
            new_grammar1.append(split_grammar1[i])

        # past indices
        for i in range(xover_idx2, len(split_grammar2)):
            new_grammar1.append(split_grammar2[i])

    else:  # one of the genomes was length 1
        new_grammar1 = []

        if len(split_grammar1) == 1:
            new_grammar1 = copy.deepcopy(split_grammar2)
            new_grammar1.insert(ind1.rng.randint(0, len(split_grammar2)),
                                split_grammar1[0])
        else:
            new_grammar1 = copy.deepcopy(split_grammar1)
            new_grammar1.insert(ind1.rng.randint(0, len(split_grammar1)),
                                split_grammar2[0])

    c1.grammar = ",".join(new_grammar1)
    return c1


# And single-point mutation
def singlePointMutation(ind):
    mutator = copy.deepcopy(ind)

    # clear the canvas if the command line parameter is set
    if args.clear_canvas:
        mutator.image = Image.new("RGBA", DIM, "black")
        mutator.isEvaluated = False

    # Change a technique.
    # if random.random() < 0.25:
    if ind.rng.random() < 0.25:
        split_grammar = mutator.grammar.split(",")
        mut_idx = ind.rng.randint(0, len(split_grammar) - 1)

        # either replace with a single technique or the possibility
        # of recursive techniques
        flattener = "#technique#"
        if ind.rng.random() < 0.5:
            flattener = "#techniques#"
        local_grammar = ExperimentSettings.grammar.flatten(flattener)

        split_grammar[mut_idx] = local_grammar
        mutator.grammar = ",".join(split_grammar)
    elif ind.rng.random() < 0.9:
        # Mutate an individual technique.
        split_grammar = mutator.grammar.split(",")
        mut_idx = ind.rng.randint(0, len(split_grammar) - 1)
        #print("\tMutation Attempt:",split_grammar[mut_idx])
        gene = split_grammar[mut_idx].split(":")
        technique = gene[0]

        # these need to become embedded within the technique itself as a
        # class
        if technique == "pixel-sort":
            gene[1] = str(ind.rng.randint(0,
                                         359))  # Mutate the angle of the sort.

            # interval function
            gene[2] = ind.rng.choice(
                ['random', 'edges', 'threshold', 'waves', 'none'])
            # sorting function
            gene[3] = ind.rng.choice(
                ['lightness', 'hue', 'saturation', 'intensity', 'minimum'])
            # randomness val
            gene[4] = str(round(ind.rng.uniform(0.0, 1.0), 2))
            # lower threshold
            gene[5] = str(round(ind.rng.uniform(0.0, 0.25), 2))
            # upper threshold
            gene[6] = str(round(ind.rng.uniform(0.0, 1.0), 2))

        elif technique == "flow-field":
            gene[1] = ind.rng.choice(["edgy", "curves"])
            gene[2] = str(round(ind.rng.uniform(0.001, 0.5), 3))
        elif technique == "flow-field-2":
            gene[1] = ind.rng.choice(palettes)
            gene[2] = ind.rng.choice(["edgy", "curvy"])
            gene[3] = str(ind.rng.randint(200, 600))
            gene[4] = str(round(ind.rng.uniform(2, 5), 2))
        elif technique == "circle-packing":
            gene[1] = ind.rng.choice(palettes)
            gene[2] = str(ind.rng.randint(10, 30))
        elif technique == "rgb-shift":
            gene[1] = str(round(ind.rng.uniform(0.0, 1.0), 2))
            gene[2] = str(round(ind.rng.uniform(0.0, 1.0), 2))
            gene[3] = str(round(ind.rng.uniform(0.0, 1.0), 2))
            gene[4] = str(ind.rng.randint(-5,5))
            gene[5] = str(ind.rng.randint(-5,5))
            gene[6] = str(ind.rng.randint(-5,5))
            gene[7] = str(ind.rng.randint(-5,5))
            gene[8] = str(ind.rng.randint(-5,5))
            gene[9] = str(ind.rng.randint(-5,5))
        elif technique == "noise-map":
            gene[1] = ind.rng.choice(palettes)
            gene[2] = str(round(ind.rng.uniform(0.001, 0.25), 3))
            gene[3] = str(round(ind.rng.uniform(0.001, 0.25), 3))
            gene[4] = str(round(ind.rng.uniform(0.0, 1.0), 2))
        elif technique == "oil-painting":
            gene[1] = str(ind.rng.randint(1,64))
        elif technique == "watercolor-filter":
            gene[1] = str(ind.rng.randint(1, 20))
            gene[2] = str(round(ind.rng.uniform(0.0, 0.5), 2))
        elif technique == "pencil-filter":
            gene[1] = str(ind.rng.randint(1, 20))
            gene[2] = str(round(ind.rng.uniform(0.0, 0.5), 2))
            gene[3] = str(round(ind.rng.uniform(0.0, 0.05), 3))
            gene[4] = ind.rng.choice(["on", "off"])
        elif technique == "walkers":
            gene[1] = ind.rng.choice(palettes)
            gene[2] = str(ind.rng.randint(10, 100))
            gene[3] = ind.rng.choice(['ordered', 'random', 'rule'])
        elif technique == "basic-trig":
            gene[1] = ind.rng.choice(palettes)
            gene[2] = str(ind.rng.randint(1, 100))
            gene[3] = ind.rng.choice(['circle', 'rect'])

        # no params here - placeholders if we augment
        # elif technique == "stippled":
        #     pass
        # elif technique == "wolfram-ca":
        #     pass
        # elif technique == "drunkardsWalk":
        #     pass
        # elif technique == "dither":
        #     pass

        split_grammar[mut_idx] = ":".join(gene)
        mutator.grammar = ",".join(split_grammar)
    else:
        # Shuffle the order of techniques
        split_grammar = mutator.grammar.split(",")
        ind.rng.shuffle(split_grammar)
        mutator.grammar = ",".join(split_grammar)

    return mutator


##########################################################################################


def roulette_selection(objs, obj_wts):
    """ Select a listing of objectives based on roulette selection. """
    obj_ordering = []

    tmp_objs = objs
    tmp_wts = obj_wts

    for i in range(len(objs)):
        sel_objs = [list(a) for a in zip(tmp_objs, tmp_wts)]

        # Shuffle the objectives
        objs[0].rng.shuffle(sel_objs)

        # Generate a random number between 0 and 1.
        # ran_num = random.random()
        ran_num = objs[0].rng.random()

        # Iterate through the objectives until we select the one we want.
        for j in range(len(sel_objs)):
            if sel_objs[j][1] > ran_num:
                obj_ordering.append(sel_objs[j][0])

                # Remove the objective and weight from future calculations.
                ind = tmp_objs.index(sel_objs[j][0])

                del tmp_objs[ind]
                del tmp_wts[ind]

                # Rebalance the weights for the next go around.
                tmp_wts = [k / sum(tmp_wts) for k in tmp_wts]
            else:
                ran_num -= sel_objs[j][1]

    return obj_ordering


def select_elite(population):
    """ Select the best individual from the population by looking at the farthest distance traveled.

    Args:
        population: population of individuals to select from.

    Returns:
        The farthest traveling individual.
    """
    best_ind = population[0]
    dist = population[0].fitness.values[0]

    for ind in population[1:]:
        if ind.fitness.values[0] > dist:
            best_ind = ind
            dist = ind.fitness.values[0]

    return best_ind


def epsilon_lexicase_selection(population,
                               generation,
                               tournsize=4,
                               shuffle=True,
                               prim_shuffle=True,
                               num_objectives=0,
                               epsilon=0.9,
                               excl_indicies=[]):
    """ Implements the epsilon lexicase selection algorithm proposed by LaCava, Spector, and McPhee.

    Selects one individual from a population by performing one individual epsilon lexicase selection event.

    Args:
        population: population of individuals to select from
        generation: what generation is it (for logging)
        tournsize: tournament size for each selection
        shuffle: whether to randomly shuffle the indices
        prim_shuffle: should we shuffle the first objective with the other fit indicies
        excl_indicies: what indicies should we exclude
    Returns:
        An individual selected using the algorithm
    """
    global glob_fit_indicies

    rng = population[0].rng

    # Get the fit indicies from the global fit indicies.
    fit_indicies = glob_fit_indicies if not shuffle else [
        i for i in range(len(population[0].fitness.weights))
    ]

    # Remove excluded indicies
    if not shuffle:
        fit_indicies = [i for i in fit_indicies if i not in excl_indicies]

    # Shuffle fit indicies if passed to do so.
    if shuffle:
        # Only shuffle "secondary" objectives leaving the first objective always
        # at the forefront.
        if not prim_shuffle:
            fit_indicies = fit_indicies[1:]
            rng.shuffle(fit_indicies)
            fit_indicies = [0] + fit_indicies
        else:
            rng.shuffle(fit_indicies)

    # Limit the number of objectives as directed.
    if num_objectives != 0:
        fit_indicies = fit_indicies[:num_objectives]

    # Sample the tournsize individuals from the population for the comparison
    sel_inds = rng.sample(population, tournsize)

    tie = True

    # Now that we have the indicies, perform the actual lexicase selection.
    # Using a threshold of epsilon (tied if within epsilon of performance)
    for k, fi in enumerate(fit_indicies):
        # Figure out if this is a minimization or maximization problem.
        min_max = (-1 * sel_inds[0].fitness.weights[fi])

        # Rank the individuals based on fitness performance for this metric.
        # Format: fit_value,index in sel_ind,rank

        fit_ranks = [[ind.fitness.values[fi], i, -1]
                     for i, ind in enumerate(sel_inds)]
        fit_ranks = [[i[0], i[1], j] for j, i in enumerate(
            sorted(fit_ranks, key=lambda x: (min_max * x[0])))]

        # Check to see if we're within the threshold value.
        for i in range(1, len(fit_ranks)):
            if math.fabs(fit_ranks[i][0] - fit_ranks[0][0]) / (
                    fit_ranks[0][0] + 0.0000001) < (1.0 - epsilon):
                fit_ranks[i][2] = fit_ranks[0][2]

        # Check to see if we have ties.
        for i in range(1, len(fit_ranks)):
            if fit_ranks[0][2] == fit_ranks[i][2]:
                tie = True
                tie_index = i + 1
            elif i == 1:
                tie = False
                break
            else:
                tie_index = i
                break

        if tie:
            sel_inds = [sel_inds[i[1]] for i in fit_ranks[:tie_index]]
            Logging.lexicase_information.append(
                [generation, k, fi, [ind._id for ind in sel_inds], -1])
        else:
            selected_individual = sel_inds[fit_ranks[0][1]]
            Logging.lexicase_information.append([
                generation, k, fi, [ind._id for ind in sel_inds],
                selected_individual._id
            ])
            tie = False
            break

    # If tie is True, we haven't selected an individual as we've reached a tie state.
    # Select randomly from the remaining individuals in that case.
    if tie:
        selected_individual = rng.choice(sel_inds)
        Logging.lexicase_information.append([
            generation, -1, -1, [ind._id for ind in sel_inds],
            selected_individual._id
        ])

    return selected_individual


##########################################################################################


def shuffle_fit_indicies(individual, excl_indicies=[]):
    """ Shuffle the fitness indicies and record them in the lexicase log. 
    
    Args:
        individual: pass one individual so we can get the fitness objectives
        excl_indicies: fitness objectives that are not under selective consideration
    """

    global glob_fit_indicies

    # Get the fitness indicies assigned to an individual
    fit_indicies = [i for i in range(len(individual.fitness.weights))]

    # Remove excluded indicies
    fit_indicies = [i for i in fit_indicies if i not in excl_indicies]

    individual.rng.shuffle(fit_indicies)

    glob_fit_indicies = fit_indicies
