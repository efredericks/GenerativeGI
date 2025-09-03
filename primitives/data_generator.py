import random

from PIL import Image, ImageDraw, ImageChops

# Draw an image 64x64 with a randomly generated grayscale background
def generate_image(circle_upper_left=False):
    """Generates a 64x64 grayscale image with random shapes.
    
    Args:
        circle_upper_left (bool): If True, draws a circle in the upper left quadrant and no circles elsewhere.
    """
    # Create a new image with a white background
    img = Image.new('L', (64, 64), color=255)
    
    # Create a draw object
    draw = ImageDraw.Draw(img)

    

    # Draw random shapes on the image
    for _ in range(15):  # Draw 15 random shapes
        x1 = random.randint(0, 63)
        y1 = random.randint(0, 63)
        x2 = random.randint(x1, 63)
        y2 = random.randint(y1, 63)


        if (j := random.randint(0, 4)) == 0:
            # Draw a rectangle with a random grayscale value
            draw.rectangle([x1, y1, x2, y2], fill=random.randint(0, 255))
        elif j == 1:
            # Draw a circle with a random grayscale value not in the upper left quadrant
            radius = random.randint(4, 20)
            center_x = random.randint(radius, 63 - radius) if radius < 63-radius else random.randint(63-radius, radius)
            low = radius if radius <= 63-radius else 63-radius
            high = 63 - radius if radius < 63-radius else radius
            center_y = random.randint(low, high) 
            # Ensure the circle is not in the upper left quadrant
            if center_x < 32:
                low = 32 + radius if 32 + radius <= 63-radius else 63-radius
                high = 63 - radius if 32 + radius < 63-radius else 32+radius
                center_y = random.randint(low, high)
            draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius], fill=random.randint(0, 255))
        elif j == 2:
            # Draw a hexagon with a random grayscale value
            hexagon_points = [
                (random.randint(0, 63), random.randint(0, 63)),
                (random.randint(0, 63), random.randint(0, 63)),
                (random.randint(0, 63), random.randint(0, 63)),
                (random.randint(0, 63), random.randint(0, 63)),
                (random.randint(0, 63), random.randint(0, 63)),
                (random.randint(0, 63), random.randint(0, 63))
            ]
            draw.polygon(hexagon_points, fill=random.randint(0, 255))
        else:
            # Draw a triangle with a random grayscale value
            triangle_points = [
                (random.randint(0, 63), random.randint(0, 63)),
                (random.randint(0, 63), random.randint(0, 63)),
                (random.randint(0, 63), random.randint(0, 63))
            ]
            draw.polygon(triangle_points, fill=random.randint(0, 255))

    # If circle_upper_left is True, draw a circle in the upper left quadrant
    if circle_upper_left:
        radius = random.randint(4, 20)
        center_x = random.randint(radius, 31 - radius) if radius <= 31-radius else random.randint(31-radius, radius)
        center_y = random.randint(radius, 31 - radius) if radius <= 31-radius else random.randint(31-radius, radius)
        draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius], fill=random.randint(0, 235))

    return img

# Generate and save the image in a folder named test_images
if __name__ == "__main__":
    # Generate 10,000 test images with circles in the upper left quadrant
    for i in range(10000):
        img = generate_image(circle_upper_left=True)
        img.save(f"test_images/circ_upp_left_generated_image_{i}.png")

    # Generate 10,000 test images without circles in the upper left quadrant
    for i in range(10000):
        img = generate_image(circle_upper_left=False)
        img.save(f"test_images/no_circ_upp_left_generated_image_{i}.png")