from PIL import Image, ImageDraw
from techniques import *
from time import sleep

DIM = (1000,1000)
background = "black"

if __name__ == "__main__":
    image = Image.new("RGBA", DIM, background)

    stippledBG(image, "red", DIM)
    image.save("temp.png")
    image = simpleDither(image)
    image.save("temp.dither.png")