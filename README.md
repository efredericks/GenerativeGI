# GenerativeGI

Genetic algorithm / novelty search for discovering generative artwork via evolving a grammar-based solution.

## Usage

I used Python 3.8, though it should work for most current versions of Python.

1. Install required libraries: `python3 -m pip install -r requirements.txt`

2. Run: `python3 main.py [args]`

  * Arguments:
    
     * Help: `--h`

     * Number of generations: `--generations [int]`

     * Population size: `--population size [int]`

     * Crossover rate: `--crossover_rate [float: [0.0, 1.0]]`

     * Mutation rate: `--mutation_rate [float: [0.0, 1.0]]`

## Grammar construction

TBD

## Adding a new technique

To add a technique for drawing you just have to create a self-contained function that accepts a PIL image as input and either updates that same image or returns a new one (up to you, I have some mixed ways to handle that I think).

Techniques are called via the grammar (in the main function in `main.py`).  Basically, you would just add a new rule to the technique rule (with a colon after - I use that for splitting parameters).

Then if you wanted parameters, you can add them below as a new rule.  Currently the only one that accepts parameters is pixel sort, however I'll expand the flow field in the near future to be different.  Note that, if you want to call a different rule, you need to surround it with a pound sign (so for instance, #technique# would be filled with any of the rules within the technique rule) - Tracery in Python isn't too bad, here's a good ref if you want to play with it at all: https://www.brettwitty.net/tracery-in-python.html

So for instance, if I wanted to add circle packing without parameters (just for testing), I'd do something like this:

In `techniques.py`:

Add:

```python
def circlePacking(img):
  # perform circle packing on img directly
```

In `main.py`:

Update the grammar rules:
```python
rules = {
...
technique = ['stippled:',
             ...
             'circlePacking:'],
}
```

Then at the top within the `evaluate` function:

```python
def evaluate(g): 
   ...
   elif _technique[0] == 'circlePacking':
       circlePacking(g.image)
```

## Techniques

This section outlines the implemented techniques, their parameters, and how to call them via the grammar.

### Flow Field

**Parameters**

* Particle size
* Particle lifetime
* Zoom level (multX, multY)
* Palette
* Style
  * Edgy
  * Flowy

#### Grammar Specification

TBD

### Stipple

TBD

#### Grammar Specification

TBD

### Pixel Sort

TBD

#### Grammar Specification

TBD