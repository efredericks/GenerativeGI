# Gets average time of each technique for GPTP 

from PIL import Image, ImageDraw
from techniques import *
from time import sleep
from colour_palettes import palettes

from evol_utils import evaluate_individual, initIndividual
from generative_object import GenerativeObject

import random, math


DIM = (500,500)#(1000, 1000)
BACKGROUND = 'black'

def runGrammar(_grammar, filename, rng):

    g = GenerativeObject(DIM, rng, _grammar)
    # g.grammar = _grammar
    g = evaluate_individual(g)
    g.image.save(filename)

rules_to_time = [
    'stippled', 
    'wolfram-ca',
    'flow-field',
    'pixel-sort',
    'drunkardsWalk', 
    'dither',
    'flow-field-2',
    'circle-packing',
    'rgb-shift',
    'noise-map',
    'oil-painting-filter',
    'watercolor-filter',
    'pencil-filter',
    'walkers',
    'basic_trig',
]

### This is a copy from settings.py to keep it pristine - if the grammar changes and this needs to be run again then it should be updated.
test_rules = {
    'ordered_pattern': ['#technique#'],
    'technique': [
        'stippled:', 'wolfram-ca:#palette#',
        'flow-field:#flow-field-type#:#flow-field-zoom#',
        'pixel-sort:#pixel-sort-angle#:#pixel-sort-interval#:#pixel-sort-sorting#:#pixel-sort-randomness#:#pixel-sort-charlength#:#pixel-sort-lowerthreshold#:#pixel-sort-upperthreshold#',
        'drunkardsWalk:#palette#', 'dither:#ditherType#',
        'flow-field-2:#palette#:#flow-field-2-type#:#flow-field-2-noisescale#:#flow-field-2-resolution#',
        'circle-packing:#palette#:#circle-packing-limit#',
        'rgb-shift:#alphaR#:#alphaG#:#alphaB#:#rXoff#:#rYoff#:#gXoff#:#gYoff#:#bXoff#:#bYoff#',
        'noise-map:#palette#:#noiseX#:#noiseY#:#noiseAlpha#',
        'oil-painting-filter:#oil-dynratio#',
        'watercolor-filter:#wc-sigma_s#:#wc-sigma_r#',
        'pencil-filter:#p-sigma_s#:#p-sigma_r#:#p-shade_factor#:#p-isbw#',
        'walkers:#palette#:#num_walkers#:#walk_type#',
        'basic_trig:#palette#:#trig_num_to_draw#:#trig_draw_type#',
    ],
    # pixel sort parameters
    'pixel-sort-angle': [str(x) for x in range(0, 360)],
    'pixel-sort-interval': ['random', 'edges', 'threshold', 'waves', 'none'],
    'pixel-sort-sorting':
    ['lightness', 'hue', 'saturation', 'intensity', 'minimum'],
    'pixel-sort-randomness': [str(x) for x in np.arange(0.0, 1.0, 0.05)],
    'pixel-sort-charlength': [str(x) for x in range(1, 30)],
    'pixel-sort-lowerthreshold': [str(x) for x in np.arange(0.0, 0.25, 0.01)],
    'pixel-sort-upperthreshold': [str(x) for x in np.arange(0.0, 1.0, 0.01)],
    # flow field parameters
    'flow-field-type': ['edgy', 'curves'],
    'flow-field-zoom': [str(x) for x in np.arange(0.001, 0.5, 0.001)],
    # flow field v2 parameters
    'flow-field-2-type': ['edgy','curvy'],
    'flow-field-2-noisescale': [str(x) for x in range(200, 600)],
    'flow-field-2-resolution': [str(x) for x in range(2, 5)],
    # circle packing parameters
    'circle-packing-limit': [str(x) for x in range(10, 30)],
    # colour palettes
    'palette': [x for x in palettes],
    # dither parameters
    'ditherType': ['grayscale', 'halftone', 'dither', 'primaryColors', 'simpleDither'],
    # rgb shift parameters
    'alphaR': [str(x) for x in np.arange(0.0, 1.0, 0.01)], 
    'alphaG': [str(x) for x in np.arange(0.0, 1.0, 0.01)],
    'alphaB': [str(x) for x in  np.arange(0.0, 1.0, 0.01)],
    'rXoff': [str(x) for x in range(-5,5)],
    'rYoff': [str(x) for x in range(-5,5)],
    'gXoff': [str(x) for x in range(-5,5)],
    'gYoff': [str(x) for x in range(-5,5)],
    'bXoff': [str(x) for x in range(-5,5)],
    'bYoff': [str(x) for x in range(-5,5)],
    # noisemap parameters
    'noiseX': [str(x) for x in np.arange(0.001, 0.25, 0.001)],
    'noiseY': [str(x) for x in np.arange(0.001, 0.25, 0.001)],
    'noiseAlpha': [str(x) for x in np.arange(0.0, 1.0, 0.01)],
    # opencv filter parameters
    'oil-dynratio': [str(x) for x in range(1,64)],
    'wc-sigma_s': [str(x) for x in range(1,20)],#200)],
    'wc-sigma_r': [str(x) for x in np.arange(0.0, 0.5, 0.01)],
    'p-sigma_s': [str(x) for x in range(1,20)],#0)],
    # 'p-sigma_s': [str(x) for x in range(1,2020)],#0)],
    'p-sigma_r': [str(x) for x in np.arange(0.0, 0.5, 0.01)],
    'p-shade_factor': [str(x) for x in np.arange(0.0, 0.05, 0.001)],
    'p-isbw': ['on', 'off'],
    # walkers
    'num_walkers': [str(x) for x in range(10,100)],
    'walk_type': ['ordered', 'random', 'rule'],
    # basic trig functions
    'trig_num_to_draw': [str(x) for x in range(1,100)],
    'trig_draw_type': ['circle', 'rect'],
}


rng = random.Random(0)
test_grammar = tracery.Grammar(test_rules)
num_replicates = 25
for rtt in rules_to_time:
    remaining = num_replicates
    while remaining > 0:
        tg = test_grammar.flatten("#technique#")
        if tg.split(':')[0].strip() == rtt:
            idx = f"{rtt}_{remaining}"
            print(tg)
            runGrammar(tg, f"./technique_time_outputs/{idx}.png", rng)
            remaining -= 1