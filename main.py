# FF:
#   FF1: maximize diversity on canvas
#   FF2: minimize size of generated code
from PIL import Image, ImageDraw, ImageChops
import opensimplex
import random
import math
from generative_object import GenerativeObject

DIM = (1000,1000)

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

def stippledBG(draw, fill):
    for y in range(DIM[1]):
        num = int(DIM[0] * p5map(y,0,DIM[1],0.01,0.2))
        for _ in range(num):
            x = random.randint(0,DIM[0]-1)
            draw.point((x,y), fill)

# Noise from opensimplex.noise returns [-1,1]
def flowField(draw, cellsize, numrows, numcols, fill, multX=0.01, multY=0.01):
    grid = []
    for r in range(numrows):
        grid.append([])
        for c in range(numcols):
            n = opensimplex.noise2(x=c*multX,y=r*multY)
            grid[r].append(p5map(n,-1.0, 1.0, 0.0, 2.0*math.pi))

    particles = []
    for _ in range(1000):
        p = {'x': random.randint(0,numcols-1), 'y': random.randint(0,numrows-1)}
        particles.append(p)

    while len(particles) > 0:
        #print(len(particles))
        for i in range(len(particles)-1, -1, -1):
            p = particles[i]
            draw.point((p['x'], p['y']), fill)

            angle = grid[int(p['y'])][int(p['x'])]

            p['x'] += math.cos(angle)
            p['y'] += math.sin(angle)
           # print(p)

            if (p['x'] < 0 or p['x'] > numcols-1 or p['y'] < 0 or p['y'] > numrows-1):
                particles.pop(i)
    return

if __name__ == "__main__":
    #img = Image.new("RGBA", DIM, "grey")
    #draw = ImageDraw.Draw(img)

    #img2 = Image.new("RGBA", DIM, "grey")
    #draw2 = ImageDraw.Draw(img2)


    #fill = (0,0,0)

    opensimplex.seed(random.randint(0,100000))
    #flowField(draw, 1, DIM[1], DIM[0])
    #stippledBG(draw, fill)
    #print(rmsdiff(img, img2), rmsdiff(img, img3), rmsdiff(img2, img3))

    num_gens = 10
    pop_size = 10
    population = []

    for gen in range(num_gens):
        g = GenerativeObject(DIM)

        if random.random() > 0.5:
            flowField(g.draw, 1, DIM[1], DIM[0], (255,0,255))
        else:
            stippledBG(g.draw, (255,0,255))

        g.image.show()
        population.append(g)

        population.clear()
    print("Done")

