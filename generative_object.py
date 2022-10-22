from PIL import Image, ImageDraw
import opensimplex
import random
import math

class GenerativeObject:
  def __init__(self, dim, background="black"):
    self.dim = dim
    self.image = Image.new("RGBA", dim, background)
    self.draw = ImageDraw.Draw(self.image)

      
