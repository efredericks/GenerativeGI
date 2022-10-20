# FF:
#   FF1: maximize diversity on canvas
#   FF2: minimize size of generated code
from PIL import Image, ImageDraw, ImageChops
import random
import math

DIM = (400,400)

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

if __name__ == "__main__":
    #output = "polygons.png"
    img = Image.new("RGBA", DIM, "grey")
    draw = ImageDraw.Draw(img)
    img2 = Image.new("RGBA", DIM, "grey")
    draw2 = ImageDraw.Draw(img2)

    fill = (0,0,0)

    stippledBG(draw, fill)
    stippledBG(draw2, (255,0,255))

    img3 = Image.new("RGBA", DIM, "grey")
    draw3 = ImageDraw.Draw(img3)
    for i in range(100000):
        draw3.point((random.randint(0,DIM[0]-1), random.randint(0,DIM[1]-1)), (random.randint(0,255), random.randint(0,255), random.randint(0,255)))

    #diff = ImageChops.difference(img, img2)
    diff = rmsdiff(img, img2)
    print(rmsdiff(img, img2), rmsdiff(img, img3), rmsdiff(img2, img3))
    img3.show()
    #img.show()
    #img2.show()
    #diff.show()

    #draw.polygon(((100, 100), (200, 50), (125, 25)), fill="green")
    #draw.polygon(((175, 100), (225, 50), (200, 25)), outline="yellow")
    #img.save(output)


    #recording for S1
