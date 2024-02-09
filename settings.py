import tracery
import numpy as np
from colour_palettes import palettes

# tbd: palettes in other techniques!

DIM = (500,500)#(1000, 1000)
BACKGROUND = 'black'

# tracery grammar
# leave a trailing colon after each technique for the parameter list as we're splitting on colon regardless
rules = {
    'ordered_pattern': ['#techniques#'],
    'techniques': ['#technique#', '#techniques#,#technique#'],
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
grammar = tracery.Grammar(rules)
