from PIL import Image, ImageOps
import numpy as np

class DataProcessing:
    """ Processing on the image """

    def __init__(self):
        pass


    """ Data convert to greyscale"""
    def img_convert_to_greyscale(self,image):

        # Don't change the already grayscale images
        if(image.ndim == 2):
            return image

        pil_image = Image.fromarray(image)

        grayscale_image = pil_image.convert('L')

        return np.array(grayscale_image)

    """ Image normalization """
    def img_normalize(self,image):
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=-1)
        return image