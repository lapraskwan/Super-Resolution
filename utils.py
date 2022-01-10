import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from custom_image_dataset_from_directory import image_dataset_from_directory
import matplotlib.pyplot as plt
import os
from PIL import Image

def upscale_image(image, upscale_factor):
    return image.resize((image.size[0] * upscale_factor, image.size[1] * upscale_factor), Image.BICUBIC)
    
def downscale_image(image, upscale_factor):
    return image.resize((image.size[0] // upscale_factor, image.size[1] // upscale_factor), Image.BICUBIC)

def image_array_from_directory(dir, downscale_factor=1, downscale=False, limit=None):
    data = []

    files = sorted(os.listdir(dir))

    if limit is None:
        limit = len(files)

    for filename in files:
        image = tf.keras.preprocessing.image.load_img(os.path.join(dir, filename))
        # Crop image so that its dimensions are divisible by the downscale factor
        image = image.resize((image.size[0] - image.size[0] % downscale_factor, image.size[1] - image.size[1] % downscale_factor))
        if downscale:
            image = downscale_image(image, downscale_factor)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image /= 255
        data.append(image)
        if len(data) >= limit:
            break

    print(f"{len(data)} images {f'downscaled by a factor of {downscale_factor}' if downscale else 'loaded'} from {dir}.")
    return data

