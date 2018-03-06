from PIL import Image
import cv2
"""Uitil to handle image."""


# Deprecated!!!!!!
def resize_image_bytes(fileFullName: str) -> str:
    """Reize image function."""
    max_height = 1024
    max_width = 1024
    with Image.open(fileFullName) as img:
        width, height = img.size
        print("original size:")
        print(img.size)

        # only shrink if img is bigger than required
        if max_height < height or max_width < width:
            # get scaling factor
            scaling_factor = max_height / float(height)
            if max_width / float(width) < scaling_factor:
                scaling_factor = max_width / float(width)
                # resize image
                img = img.resize((int(width * scaling_factor),
                                  int(height * scaling_factor)),
                                 Image.ANTIALIAS)

        print("Scaled size:")
        print(img.size)

        return img.tobytes()


def resize_image(fileFullName: str) -> str:
    """Reize image function."""
    input = cv2.imread(fileFullName)

    height, width = input.shape[:2]

    print("original size:")
    print(input.shape[:2])
    max_height = 1024
    max_width = 1024

    # only shrink if img is bigger than required
    if max_height < height or max_width < width:
        # get scaling factor
        scaling_factor = max_height / float(height)
        if max_width / float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        # resize image
        input = cv2.resize(
            input,
            None,
            fx=scaling_factor,
            fy=scaling_factor,
            interpolation=cv2.INTER_AREA)

        print("caled size:")
        print(input.shape[:2])

    return input
