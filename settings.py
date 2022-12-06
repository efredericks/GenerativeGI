import tracery
import numpy as np

DIM = (1000, 1000)

# tracery grammar
# leave a trailing colon after each technique for the parameter list as we're splitting on colon regardless
rules = {
    'ordered_pattern': ['#techniques#'], 
    'techniques': ['#technique#', '#techniques#,#technique#'],
    'technique': ['stippled:', 
                'wolfram-ca:',
                'flow-field:#flow-field-type#:#flow-field-zoom#', 
                'pixel-sort:#pixel-sort-angle#:#pixel-sort-interval#:#pixel-sort-sorting#:#pixel-sort-randomness#:#pixel-sort-charlength#:#pixel-sort-lowerthreshold#:#pixel-sort-upperthreshold#', 
                'dither:'],
    # pixel sort parameters
    'pixel-sort-angle': [str(x) for x in range(0,360)],
    'pixel-sort-interval': ['random', 'edges', 'threshold', 'waves', 'none'],
    'pixel-sort-sorting': ['lightness', 'hue', 'saturation', 'intensity', 'minimum'],
    'pixel-sort-randomness': [str(x) for x in np.arange(0.0, 1.0, 0.05)],
    'pixel-sort-charlength': [str(x) for x in range(1,30)],
    'pixel-sort-lowerthreshold': [str(x) for x in np.arange(0.0, 0.25, 0.01)],
    'pixel-sort-upperthreshold': [str(x) for x in np.arange(0.0, 1.0, 0.01)],
    # flow field parameters
    'flow-field-type': ['edgy', 'curves'],
    'flow-field-zoom': [str(x) for x in np.arange(0.001, 0.5, 0.001)],
}
grammar = tracery.Grammar(rules)