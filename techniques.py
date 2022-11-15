from PIL import Image, ImageDraw, ImageChops
import opensimplex
from pixelsort import pixelsort
import random
import math
import numpy as np

### Utility functions
def p5map(n, start1, stop1, start2, stop2):
    return ((n-start1)/(stop1-start1))*(stop2-start2)+start2

# https://stackoverflow.com/questions/3098406/root-mean-square-difference-between-two-images-using-python-and-pil
def rmsdiff(im1, im2):
    "Calculate the root-mean-square difference between two images"
    diff = ImageChops.difference(im1, im2)
    h = diff.histogram()
    sq = (value*((idx%256)**2) for idx, value in enumerate(h))
    sum_of_squares = sum(sq)
    rms = math.sqrt(sum_of_squares/float(im1.size[0] * im1.size[1]))
    return rms

# c/o https://codereview.stackexchange.com/questions/55902/fastest-way-to-count-non-zero-pixels-using-python-and-pillow
def count_nonblack_pil(img):
    bbox = img.getbbox()
    if not bbox: return 0
    return sum(img.crop(bbox)
               .point(lambda x: 255 if x else 0)
               .convert("L")
               .point(bool)
               .getdata())
###

def pixelSort(img, params):#angle, interval, sorting, randomness):
    #t =['random', 'edges', 'threshold', 'waves', 'none']
    #random.shuffle(t)
    return pixelsort(img, 
        angle=int(params[0]), 
        interval_function=params[1], 
        sorting_function=params[2], 
        randomness=float(params[3]),
#        char_length=float(params[4]),
        lower_threshold=float(params[5]),
        upper_threshold=float(params[6])
    )
    """
        image: Image.Image,
        mask_image: typing.Optional[Image.Image] = None,
        interval_image: typing.Optional[Image.Image] = None,
        randomness: float = DEFAULTS["randomness"],
        char_length: float = DEFAULTS["char_length"],
        sorting_function: typing.Literal["lightness", "hue", "saturation", "intensity", "minimum"] = DEFAULTS[
            "sorting_function"],
        interval_function: typing.Literal["random", "threshold", "edges", "waves", "file", "file-edges", "none"] =
        DEFAULTS["interval_function"],
        lower_threshold: float = DEFAULTS["lower_threshold"],
        upper_threshold: float = DEFAULTS["upper_threshold"],
        angle: float = DEFAULTS["angle"]
    """

# "simple" PIL dithering
def simpleDither(img):
    dithered = img.convert(mode="1")
    dithered = dithered.convert("RGBA")
    return dithered



def stippledBG(img, fill, DIM):
    draw = ImageDraw.Draw(img)
    for y in range(DIM[1]):
        num = int(DIM[0] * p5map(y,0,DIM[1],0.01,0.2))
        for _ in range(num):
            x = random.randint(0,DIM[0]-1)
            draw.point((x,y), fill)

# Noise from opensimplex.noise returns [-1,1]
# TBD: frequency?
def flowField(img, cellsize, numrows, numcols, fill, flowType, multX=0.01, multY=0.01):
    # unpack the string
    multX = float(multX)
    multY = float(multY)

    draw = ImageDraw.Draw(img)
    grid = []
    for r in range(numrows):
        grid.append([])
        for c in range(numcols):
            n = opensimplex.noise2(x=c*multX,y=r*multY)

            if (flowType == "curves"):
                grid[r].append(p5map(n,-1.0, 1.0, 0.0, 2.0*math.pi))
            else:
                grid[r].append(math.ceil(
                  (p5map(n, 0.0, 1.0, 0.0, 2.0*math.pi) * (math.pi / 4.0)) / (math.pi / 4.0)
                ))
        

    particles = []
    for _ in range(1000):
        p = {'x': random.randint(0,numcols-1), 'y': random.randint(0,numrows-1), 'life': random.randint(numcols/2,numcols)}
        particles.append(p)

    while len(particles) > 0:
        #print(len(particles))
        for i in range(len(particles)-1, -1, -1):
            p = particles[i]
            draw.point((p['x'], p['y']), fill)

            angle = grid[int(p['y'])][int(p['x'])]

            p['x'] += math.cos(angle)
            p['y'] += math.sin(angle)
            p['life'] -= 1
           # print(p)

            if (p['x'] < 0 or p['x'] > numcols-1 or p['y'] < 0 or p['y'] > numrows-1 or p['life'] <= 0):
                particles.pop(i)
    return
