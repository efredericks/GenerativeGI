from PIL import Image, ImageDraw, ImageChops, ImageColor

# import random
import math
import numpy as np
#from settings import *
import scipy.spatial
import cv2
from sklearn.cluster import KMeans


### Utility functions
# map function similar to p5.js
def p5map(n, start1, stop1, start2, stop2):
    return ((n - start1) / (stop1 - start1)) * (stop2 - start2) + start2


# constrain value to range
def constrain(val, min_val, max_val):
    return min(max_val, max(min_val, val))


# https://stackoverflow.com/questions/3098406/root-mean-square-difference-between-two-images-using-python-and-pil
def rmsdiff(im1, im2):
    "Calculate the root-mean-square difference between two images"
    diff = ImageChops.difference(im1, im2)
    h = diff.histogram()
    sq = (value * ((idx % 256)**2) for idx, value in enumerate(h))
    sum_of_squares = sum(sq)
    rms = math.sqrt(sum_of_squares / float(im1.size[0] * im1.size[1]))
    return rms


# c/o https://codereview.stackexchange.com/questions/55902/fastest-way-to-count-non-zero-pixels-using-python-and-pillow
def count_nonblack_pil(img):
    bbox = img.getbbox()
    if not bbox: return 0
    return sum(
        img.crop(bbox).point(lambda x: 255
                             if x else 0).convert("L").point(bool).getdata())


###


def drawRectangle(img, x, y, x2, y2, fill):
    """ 
    Draw a rectangle on the image at position (x, y) with given width, height, and fill color.

    Args:
        img (PIL.Image): The image to draw on.
        x (int): The x-coordinate of the top-left corner of the rectangle.
        y (int): The y-coordinate of the top-left corner of the rectangle.
        x2 (int): The x-coordinate of the bottom-right corner of the rectangle.
        y2 (int): The y-coordinate of the bottom-right corner of the rectangle.
        fill (int): The fill color of the rectangle, range of 0 - 255.
    """
    draw = ImageDraw.Draw(img)

    # ensure x2 > x and y2 > y
    if x2 < x:
        x, x2 = x2, x
    if y2 < y:
        y, y2 = y2, y
    draw.rectangle([x, y, x2, y2], fill=(fill))
    return

def drawCircle(img, centerX, centerY, radius, fill):
    """ 
    Draw a circle on the image at position (centerX, centerY) with given radius and fill color.

    Args:
        img (PIL.Image): The image to draw on.
        centerX (int): The x-coordinate of the center of the circle.
        centerY (int): The y-coordinate of the center of the circle.
        radius (int): The radius of the circle.
        fill (int): The fill color of the circle, range of 0 - 255.
    """
    draw = ImageDraw.Draw(img)
    print(f"\t\t\t\t\t\t\t{fill}")
    draw.ellipse([centerX - radius, centerY - radius, centerX + radius, centerY + radius], fill=(fill))
    return

def drawHexagon(img, points, fill):
    """ 
    Draw a hexagon on the image with given points and fill color.

    Args:
        img (PIL.Image): The image to draw on.
        points (list): A list of tuples representing the vertices of the hexagon.
        fill (int): The fill color of the hexagon, range of 0 - 255.
    """
    draw = ImageDraw.Draw(img)
    draw.polygon(points, fill=(fill))
    return

def drawTriangle(img, points, fill):
    """ 
    Draw a triangle on the image with given points and fill color.

    Args:
        img (PIL.Image): The image to draw on.
        points (list): A list of tuples representing the vertices of the triangle.
        fill (int): The fill color of the triangle, range of 0 - 255.
    """
    draw = ImageDraw.Draw(img)
    draw.polygon(points, fill=(fill))
    return
