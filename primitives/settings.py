import tracery
import numpy as np

# tbd: palettes in other techniques!

DIM = (64, 64)
# DIM = (500,500)
BACKGROUND = 'black'

# tracery grammar
# leave a trailing colon after each technique for the parameter list as we're splitting on colon regardless
rules = {
    'ordered_pattern': ['#program#'],
    'program': ['#program#,#action#','#action#'],
    'action': ['#background#', '#primitive#', '#technique#'],
    'background': [f'background:#DIM0#:#DIM1#:#color#'],
    'primitive': [
        'point:#x#:#y#:#stroke#',
        'ellipse:#x#:#y#:#w#:#h#:#fill#:#stroke#',
        'rect:#x#:#y#:#w#:#h#:#fill#:#stroke#',
        'line:#x#:#y#:#x#:#y#:#stroke#:#strokeWeight#',
    ],
    'stroke': ['#color#'],
    'fill': ['#color#'],
    'strokeWeight': [str(x) for x in range(1, DIM[0])],#str(f"{x:.3f}") for x in np.arange(0.01, 10.0, 0.001)],
    'DIM0': str(DIM[0]),
    'DIM1': str(DIM[1]),
    'x': [str(x) for x in range(0,DIM[0])],
    'y': [str(x) for x in range(0,DIM[1])],
    'w': [str(x) for x in range(0,DIM[0])],
    'h': [str(x) for x in range(0,DIM[1])],
    'color': ['#r#:#g#:#b#:#a#'],
    'r': [str(x) for x in range(0,255)],
    'g': [str(x) for x in range(0,255)],
    'b': [str(x) for x in range(0,255)],
    'a': [str(x) for x in range(0,255)],
    # 'ordered_pattern': ['#techniques#'],
    # 'techniques': ['#technique#', '#techniques#,#technique#'],
    'technique': [
        'rectangle:#rectangle-x#:#rectangle-y#:#rectangle-x2#:#rectangle-y2#:#rectangle-fill#', 
        'circle:#circle-centerX#:#circle-centerY#:#circle-radius#:#circle-fill#', 
        'hexagon:#hexagon-point-x#:#hexagon-point-y#:#hexagon-point-x2#:#hexagon-point-y2#:#hexagon-point-x3#:#hexagon-point-y3#:#hexagon-point-x4#:#hexagon-point-y4#:#hexagon-point-x5#:#hexagon-point-y5#:#hexagon-point-x6#:#hexagon-point-y6#:#hexagon-fill#',
        'triangle:#triangle-point-x#:#triangle-point-y#:#triangle-point-x2#:#triangle-point-y2#:#triangle-point-x3#:#triangle-point-y3#:#triangle-fill#',
    ],
    # rectangle parameters
        'rectangle-x': [str(x) for x in range(0, DIM[0])],
        'rectangle-y': [str(x) for x in range(0, DIM[1])],
        'rectangle-x2': [str(x) for x in range(0, DIM[0])],
        'rectangle-y2': [str(x) for x in range(0, DIM[1])],
        'rectangle-fill': '#color#',
    # circle parameters
        'circle-centerX': [str(x) for x in range(0, DIM[0])],
        'circle-centerY': [str(x) for x in range(0, DIM[1])],
        'circle-radius': [str(x) for x in range(1, DIM[0]//4)],
        'circle-fill': '#color#',
    # hexagon parameters
        'hexagon-point-x': [str(x) for x in range(0, DIM[0])],
        'hexagon-point-y': [str(x) for x in range(0, DIM[1])],
        'hexagon-point-x2': [str(x) for x in range(0, DIM[0])],
        'hexagon-point-y2': [str(x) for x in range(0, DIM[1])],
        'hexagon-point-x3': [str(x) for x in range(0, DIM[0])],
        'hexagon-point-y3': [str(x) for x in range(0, DIM[1])],
        'hexagon-point-x4': [str(x) for x in range(0, DIM[0])],
        'hexagon-point-y4': [str(x) for x in range(0, DIM[1])],
        'hexagon-point-x5': [str(x) for x in range(0, DIM[0])],
        'hexagon-point-y5': [str(x) for x in range(0, DIM[1])],
        'hexagon-point-x6': [str(x) for x in range(0, DIM[0])],
        'hexagon-point-y6': [str(x) for x in range(0, DIM[1])],
        'hexagon-fill': '#color#',
    # triangle parameters
        'triangle-point-x': [str(x) for x in range(0, DIM[0])],
        'triangle-point-y': [str(x) for x in range(0, DIM[1])],
        'triangle-point-x2': [str(x) for x in range(0, DIM[0])],
        'triangle-point-y2': [str(x) for x in range(0, DIM[1])],
        'triangle-point-x3': [str(x) for x in range(0, DIM[0])],
        'triangle-point-y3': [str(x) for x in range(0, DIM[1])],
        'triangle-fill': '#color#',
}
grammar = tracery.Grammar(rules)
# print(grammar.flatten("#ordered_pattern#"))