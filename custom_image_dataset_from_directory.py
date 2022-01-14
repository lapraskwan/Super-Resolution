# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras image dataset loading utilities."""

""" 
I want the images to be cropped into sub-images of image_size, rather than crop it using tf.image.resize() or keras_image_ops.smart_resize(), 
so I copied the source code here and modified paths_and_labels_to_dataset() and load_image(). There is a new function crop_to_sub_images() for 
croppping out the sub-images. I also changed the default value of label_mode to None because it is useless after my modifications. 

Original file: https://github.com/keras-team/keras/blob/v2.7.0/keras/preprocessing/image_dataset.py#L30-L227 
Commit on August 15, 2021, a23d4ed 
"""

# pylint: disable=g-classes-have-attributes


from multiprocessing import Value
import tensorflow.compat.v2 as tf
import os
import numpy as np
from keras.layers.preprocessing import image_preprocessing
from keras.preprocessing import dataset_utils
from keras.preprocessing import image as keras_image_ops
from tensorflow.python.util.tf_export import keras_export
ALLOWLIST_FORMATS = ('.bmp', '.gif', '.jpeg', '.jpg', '.png')


@keras_export('keras.utils.image_dataset_from_directory',
              'keras.preprocessing.image_dataset_from_directory',
              v1=[])
def image_dataset_from_directory(directory,
                                 label_directory=None,  # I added this to support using images as label
                                 color_mode='rgb',
                                 batch_size=32,
                                 image_size=(256, 256),
                                 shuffle=True,
                                 seed=None,
                                 validation_split=None,
                                 subset=None,
                                 interpolation='bilinear',
                                 follow_links=False,
                                 stride=None,
                                 num_images=None,
                                 upscale_factor=1,
                                 num_sub_images=None,
                                 **kwargs):
    """Generates a `tf.data.Dataset` from image files in a directory.

    WARNING: If label_directory is not given, but num_sub_images is specified, the position where sub images are cropped will be random.
    Provide label_directory if you want to use num_sub_images and want the positions of sub_images in images and labels to be the same.

    Supported image formats: jpeg, png, bmp, gif.
    Animated gifs are truncated to the first frame.
    Args:
      directory: Directory where the data is located.
          If `labels` is "inferred", it should contain
          subdirectories, each containing images for a class.
          Otherwise, the directory structure is ignored.
      label_directory: Directory where the labels are located, number of file in this directory
          should be the same as the number of data in the main directory. The alphabetical order of labels 
          should be the same as that of images.
      color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
          Whether the images will be converted to
          have 1, 3, or 4 channels.
      batch_size: Size of the batches of data. Default: 32.
      image_size: Size to resize images to after they are read from disk.
          Defaults to `(256, 256)`.
          Since the pipeline processes batches of images that must all have
          the same size, this must be provided.
      shuffle: Whether to shuffle the data. Default: True.
          If set to False, sorts the data in alphanumeric order.
      seed: Optional random seed for shuffling and transformations.
      validation_split: Optional float between 0 and 1,
          fraction of data to reserve for validation.
      subset: One of "training" or "validation".
          Only used if `validation_split` is set.
      interpolation: String, the interpolation method used when resizing images.
        Defaults to `bilinear`. Supports `bilinear`, `nearest`, `bicubic`,
        `area`, `lanczos3`, `lanczos5`, `gaussian`, `mitchellcubic`.
      follow_links: Whether to visits subdirectories pointed to by symlinks.
          Defaults to False.
      stride: A (height, width) tuple representing the number of pixels between sub images. 
            If None then it is set to 2/3 of image_size to allow some overlappings.
      num_images: Number of images to be used from the directory
      upscale_factor: if the image is a label that is the high resolution version, then it will be the ratio that the image is upscaled.
      num_sub_images: number of sub-images to be cropped out from an image. If None, then the sub-images will be cropped through the whole image with the specified stride. 
            Otherwise sub-images will be randomly cropped out.
      **kwargs: Legacy keyword arguments.
    Returns:
      A `tf.data.Dataset` object.
        - If `label_directory` is None, it yields `float32` tensors of shape
          `(batch_size, image_size[0], image_size[1], num_channels)`,
          encoding images (see below for rules regarding `num_channels`).
        - Otherwise, it yields a tuple `(images, labels)`, where `images` and `labels`
          have shape `(batch_size, image_size[0], image_size[1], num_channels)`
    Rules regarding number of channels in the yielded images:
      - if `color_mode` is `grayscale`,
        there's 1 channel in the image tensors.
      - if `color_mode` is `rgb`,
        there are 3 channel in the image tensors.
      - if `color_mode` is `rgba`,
        there are 4 channel in the image tensors.
    """
    if kwargs:
        raise TypeError(f'Unknown keywords argument(s): {tuple(kwargs.keys())}')
    if color_mode == 'rgb':
        num_channels = 3
    elif color_mode == 'rgba':
        num_channels = 4
    elif color_mode == 'grayscale':
        num_channels = 1
    else:
        raise ValueError(
            '`color_mode` must be one of {"rgb", "rgba", "grayscale"}. '
            f'Received: color_mode={color_mode}')
    interpolation = image_preprocessing.get_interpolation(interpolation)
    dataset_utils.check_validation_split_arg(
        validation_split, subset, shuffle, seed)
    if seed is None:
        seed = np.random.randint(1e6)

    if label_directory is not None and upscale_factor is None:
        raise ValueError("Upscale_factor must be specified if label_directory is provided.")

    image_paths = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for file in sorted(filenames):
            image_paths.append(os.path.join(directory, file))
    
    label_paths = []
    if label_directory is not None:
        for dirpath, dirnames, filenames in os.walk(label_directory):
            for file in sorted(filenames):
                label_paths.append(os.path.join(label_directory, file))

    image_paths, _ = dataset_utils.get_training_or_validation_split(
        image_paths, None, validation_split, subset)
    label_paths, _ = dataset_utils.get_training_or_validation_split(
        label_paths, None, validation_split, subset)

    if not image_paths:
        raise ValueError(f'No images found in directory {directory}. '
                         f'Allowed formats: {ALLOWLIST_FORMATS}')
    if label_directory is not None and not label_paths:
        raise ValueError(f'No images found in directory {label_directory}. '
                         f'Allowed formats: {ALLOWLIST_FORMATS}')

    if num_images is not None:
        if shuffle:
            indices = np.random.choice(len(image_paths), num_images, replace=False)
            image_paths = [image_paths[i] for i in indices]
            label_paths = [label_paths[i] for i in indices]
        else:
            image_paths = image_paths[:num_images]
            label_paths = label_paths[:num_images]

    dataset = paths_and_labels_to_dataset(
        image_paths=image_paths,
        label_paths=label_paths,
        image_size=image_size,
        num_channels=num_channels,
        interpolation=interpolation,
        shuffle=shuffle,
        seed=seed,
        batch_size=batch_size,
        stride=stride,
        upscale_factor=upscale_factor,
        num_sub_images=num_sub_images)

    return dataset


def paths_and_labels_to_dataset(image_paths,
                                label_paths,
                                image_size,
                                num_channels,
                                interpolation,
                                shuffle,
                                seed,
                                batch_size,
                                stride=None,
                                upscale_factor=1,
                                num_sub_images=None):
    """Constructs a dataset of images and labels.
    If label_paths is empty, create a dataset with images without labels.
    If label_paths is not empty, create a dataset with images and labels.
    """
    def load_image_label_with_same_seeds(image, label, num_sub_images, image_args, label_args):
        "Ensure the sub_images of images and labels are cropped at the same position"
        sub_image_seed = tf.random.uniform((), 0, 100000, tf.int32)
        
        image_args = (*image_args, sub_image_seed)
        label_args = (*label_args, sub_image_seed)

        if num_sub_images is not None:
            image_args = (*image_args, num_sub_images)
            label_args = (*label_args, num_sub_images)

        # Use tf.py_function for using python controls like if, for
        return (tf.py_function(load_image, [image, *image_args], [tf.float32]), tf.py_function(load_image, [label, *label_args], [tf.float32]))

    image_path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    image_args = (image_size, num_channels, interpolation, stride, 1)
    if not label_paths:
        if shuffle:
            # Shuffle locally at each iteration
            image_path_ds = image_path_ds.shuffle(buffer_size=batch_size * 8, seed=seed, reshuffle_each_iteration=True)

        # Add seed to image_args
        image_args = (*image_args, tf.random.uniform((), 0, 100000, tf.int32))
        # Add num_sub_images to image_args if it is not None
        if num_sub_images is not None:
            image_args = (*image_args, num_sub_images)

        # Map path to sub-images
        img_ds = image_path_ds.map(lambda x: tf.py_function(load_image, [x, *image_args], [tf.float32]))
        # Flatten img_ds
        img_ds = img_ds.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
        # Batch
        if batch_size is not None:
            img_ds = img_ds.prefetch(tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True)
            # Set shape for the batches (Added drop_remainder=True in dataset.batch() above to ensure all batches has the same size)
            img_ds = img_ds.map(lambda batch: set_shape(batch, batch_size=batch_size, image_size=image_size, num_channels=num_channels))
    else:
        label_path_ds = tf.data.Dataset.from_tensor_slices(label_paths)
        path_ds = tf.data.Dataset.zip((image_path_ds, label_path_ds))

        hr_image_size = (image_size[0] * upscale_factor, image_size[1] * upscale_factor)
        hr_stride = (stride[0] * upscale_factor, stride[1] * upscale_factor)
        label_args = (hr_image_size, num_channels, interpolation, hr_stride, upscale_factor)

        if shuffle:
            # Shuffle locally at each iteration
            path_ds = path_ds.shuffle(buffer_size=batch_size * 8, seed=seed, reshuffle_each_iteration=True)

        # Map path to sub-images
        img_ds = path_ds.map(lambda x, y: load_image_label_with_same_seeds(x, y, num_sub_images, image_args, label_args))
        # Flatten img_ds twice, from (1, len(sub_images), height, width, channel) to (len(sub_images), height, width, channel) to (height, width, channel)
        img_ds = img_ds.flat_map(lambda x, y: tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(x), tf.data.Dataset.from_tensor_slices(y))))
        img_ds = img_ds.flat_map(lambda x, y: tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(x), tf.data.Dataset.from_tensor_slices(y))))
        # Batch
        if batch_size is not None:
            img_ds = img_ds.prefetch(tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True)
            # Set shape for the batches (Added drop_remainder=True in dataset.batch() above to ensure all batches has the same size)
            img_ds = img_ds.map(lambda image_batch, label_batch: (set_shape(image_batch, batch_size=batch_size, image_size=image_size, num_channels=num_channels),
                                                                    set_shape(label_batch, batch_size=batch_size, image_size=hr_image_size, num_channels=num_channels)))
    
    return img_ds


def load_image(path, image_size, num_channels, interpolation, stride=None, upscale_factor=1, seed=None, num_sub_images=None):
    """
    Load an image from a path and return a list of sub images.
    Return:
      A list of sub images.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=num_channels, expand_animations=False)
    
    # img = tf.image.resize(img, image_size, method=interpolation)
    img = crop_to_sub_images(img, image_size, num_channels, stride, num_sub_images, seed, upscale_factor)

    return np.array(img)


def crop_to_sub_images(img, image_size, num_channels, stride=None, num_sub_images=None, seed=None, upscale_factor=1):
    """
    Crop an image to sub images of image_size.
    Args:
        img: original image
        image_size: size of each sub image. (height, width)
        stride: A (height, width) tuple representing the number of pixels between sub images. If None then it is set to 2/3 of image_size to allow some overlappings.
        num_sub_images: number of sub-images to be cropped out from an image. If None, then the sub-images will be cropped through the whole image with the specified stride. 
                        Otherwise sub-images will be randomly cropped out.
        seed: random seed to ensure the images are cropped out from the same position.
        upscale_factor: if the image is a label that is the high resolution version, then it will be the ratio that the image is upscaled.
    Return:
        A list of sub images.
    """
    if image_size[0] > img.shape[0] or image_size[1] > img.shape[1]:
        print("Error when cropping image: image_size > original image size.")
    upper_boundary = (img.shape[0] - image_size[0], img.shape[1] - image_size[1])
    if stride is None:
        stride = (image_size[0] * 3 // 4, image_size[1] * 3 // 4)

    sub_images = []
    
    # Crop random parts of image
    if num_sub_images is not None:
        if seed is not None:
            np.random.seed(seed)
        for i in range(num_sub_images):
            start = (np.random.randint(0, upper_boundary[0] // upscale_factor) * upscale_factor, np.random.randint(0, upper_boundary[1] // upscale_factor) * upscale_factor)
            # Get the sub image at (start[0], start[1])
            sub_images.append(img[start[0]:start[0]+image_size[0], start[1]:start[1]+image_size[1], :])
        for image in sub_images:
            image.set_shape((image_size[0], image_size[1], num_channels))
        return sub_images

    # Crop whole image
    w = 0
    while True:  # Loop until width reaches upper boundary
        if w > upper_boundary[1]:
            w = upper_boundary[1]
        h = 0
        while True:  # Loop until height reaches upper boundary
            if h > upper_boundary[0]:
                h = upper_boundary[0]
            # Get the sub image at (x,y)
            sub_images.append(img[h:h+image_size[0], w:w+image_size[1], :])
            if h >= upper_boundary[0]:
                break
            h += stride[0]
        if w >= upper_boundary[1]:
            break
        w += stride[1]

    for image in sub_images:
        image.set_shape((image_size[0], image_size[1], num_channels))
    return sub_images


def set_shape(batch, batch_size, image_size, num_channels):
    batch.set_shape([batch_size, image_size[0], image_size[1], num_channels])
    return batch
