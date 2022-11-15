from PIL import Image, ImageDraw
import opensimplex
import random
import math
from techniques import *

class GenerativeObject:
  def __init__(self, idx, dim, grammar, background="black"):
    self.id = idx
    self.dim = dim
    self.grammar = grammar
    self.image = Image.new("RGBA", dim, background)
    self.isEvaluated = False
    self.fitness = 0.0

  def setFitness(self, val):
      self.fitness = val
  def getFitness(self):
      return self.fitness

  #def evaluate(self):
  #  self.isEvaluated = True
  #  for technique in self.grammar.split(','):
  #      c = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
  #      if technique == 'flow-field':
  #          flowField(self.draw, 1, self.dim[1], self.dim[0], c)
  #      elif technique == 'stippled':
  #          stippledBG(self.draw, c, self.dim)
  #  #self.image.show()

      
