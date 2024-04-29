import numpy as np
from PIL import Image, ImageDraw
import tifffile
from tqdm import tqdm
import os
import json

def find_smallest_bounding_square(binary_image):
    # Convert to grayscale if necessary
    if len(binary_image.shape) == 3:
        binary_image = np.array(Image.fromarray(binary_image).convert("L"))

    # Find indices of white pixels (masked region)
    white_pixels = np.argwhere(binary_image == 255)

    if not white_pixels.any():
        return None  # If there are no white pixels, return None

    # Find minimum and maximum row and column indices
    min_row = np.min(white_pixels[:, 0])
    max_row = np.max(white_pixels[:, 0])
    min_col = np.min(white_pixels[:, 1])
    max_col = np.max(white_pixels[:, 1])

    # Calculate width and height of bounding box
    width = max_col - min_col + 1
    height = max_row - min_row + 1

    # Determine the size of the square bounding box
    size = max(width, height)

    # Calculate coordinates of the top-left corner of the square bounding box
    
    sub = 16 if (min_col - 16 >=0 and min_row - 16 >=0 and max_row + 16 <= binary_image.shape[0] and max_col + 16 <= binary_image.shape[1]) else max(min(min_col, min_row, binary_image.shape[0] - max_row, binary_image.shape[1] - max_col), 0)

    top_left_x = min_col - sub
    top_left_y = min_row - sub

    return top_left_x, top_left_y, size + 2*sub

def draw_bb(image_path):
    """Visualize the bounding box produced by find_smallest_bounding_square

    Args:
        image_path (os.path): path to the binary image for finding bbox
    """
    # image_path = "/data/prathosh/goirik/remove/segment-anything/examples/masks/122000000009.jpg"
    binary_image = np.array(Image.open(image_path))

    bounding_square = find_smallest_bounding_square(binary_image)
    if bounding_square:
        x, y, size = bounding_square
        
        print(bounding_square)
        # Draw bounding box on the image
        image_with_bbox = Image.fromarray(binary_image.astype(np.uint8))
        draw = ImageDraw.Draw(image_with_bbox)
        draw.rectangle([x, y, x + size, y + size], outline="white")

        # Save the image with the bounding box overlaid
        image_with_bbox.save("masked.png")
        print("Image with bounding box saved as 'masked.png'.")
    else:
        pass
        print("No white pixels found in the binary image.")
        