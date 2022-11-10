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