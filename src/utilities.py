import matplotlib.pyplot as plt
from PIL import Image

def resize_and_crop_image(image, target_width, target_height):
    """
    Resize and crop an image to fit the specified width and height.

    Args:
    image (Image.Image): The original PIL image.
    target_width (int): The target width.
    target_height (int): The target height.

    Returns:
    Image.Image: The resized and cropped image.
    """
    # Calculate the target aspect ratio
    target_ratio = target_width / target_height
    # Calculate the original aspect ratio
    original_ratio = image.width / image.height

    # Determine if the image needs to be cropped by width or height
    if original_ratio > target_ratio:
        # Crop the width
        new_height = image.height
        new_width = int(target_ratio * new_height)
        left = (image.width - new_width) / 2
        top = 0
        right = left + new_width
        bottom = new_height
    else:
        # Crop the height
        new_width = image.width
        new_height = int(new_width / target_ratio)
        left = 0
        top = (image.height - new_height) / 2
        right = new_width
        bottom = top + new_height

    image = image.crop((left, top, right, bottom))
    image = image.resize((target_width, target_height), Image.LANCZOS)
    return image


def visualize_meme(image, caption):
    """
    Display a meme image along with its caption.

    Args:
    image (np.array): The image array of the meme.
    caption (str): The caption associated with the meme.
    """
    plt.figure(figsize=(8, 8))  # Set the figure size
    plt.imshow(image)  # Show the image
    plt.axis('off')  # Turn off the axis
    plt.title(caption)  # Set the caption as the title
    plt.show()
