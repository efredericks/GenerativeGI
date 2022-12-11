from PIL import Image, ImageDraw
from techniques import *
from time import sleep
from colour_palettes import palettes

DIM = (1000,1000)
background = "black"

if __name__ == "__main__":
    image = Image.new("RGBA", DIM, background)

    #drunkenWalk(image)
    #image.save("drunk.png")

    #WolframCA(image)
    #image.save("wolfram.png")

    #stippledBG(image, "red", DIM)
    #image.save("temp.png")
    #image = simpleDither(image)
    #image.save("temp.dither.png")
    
    #flowField2(image, random.choice(palettes), 'curvy', random.randrange(200, 600), random.randrange(2, 5))
    #image.save("ff.png")
    circlePacking(image, random.choice(palettes), random.randrange(10, 30))
    image.save("circles.png")
