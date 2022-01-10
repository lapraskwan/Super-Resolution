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

#### I want the images to be cropped into sub-images of image_size, ####
#### rather than crop it using tf.image.resize() or keras_image_ops.smart_resize(), ####
#### so I copied the source code here and modified paths_and_labels_to_dataset() and load_image(). ####
#### There is a new function crop_to_sub_images() for croppping out the sub-images. ####
#### I also changed the default value of label_mode to None because it is useless after my modifications. ####

#### To use the smart_resize() function, set crop_to_aspect_ratio=True. ####

#### Original file: https://github.com/keras-team/keras/blob/v2.7.0/keras/preprocessing/image_dataset.py#L30-L227 ####
##### Commit on August 15, 2021, a23d4ed ####

import tensorflow.compat.v2 as tf
# pylint: disable=g-classes-have-attributes

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
                                 labels='inferred',
                                 label_mode=None, # I changed this from 'int' to None
                                 class_names=None,
                                 color_mode='rgb',
                                 batch_size=32,
                                 image_size=(256, 256),
                                 shuffle=True,
                                 seed=None,
                                 validation_split=None,
                                 subset=None,
                                 interpolation='bilinear',
                                 follow_links=False,
                                 crop_to_aspect_ratio=False,
                                 stride=None,
                                 **kwargs):
  """Generates a `tf.data.Dataset` from image files in a directory.
  If your directory structure is:
  ```
  main_directory/
  ...class_a/
  ......a_image_1.jpg
  ......a_image_2.jpg
  ...class_b/
  ......b_image_1.jpg
  ......b_image_2.jpg
  ```
  Then calling `image_dataset_from_directory(main_directory, labels='inferred')`
  will return a `tf.data.Dataset` that yields batches of images from
  the subdirectories `class_a` and `class_b`, together with labels
  0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).
  Supported image formats: jpeg, png, bmp, gif.
  Animated gifs are truncated to the first frame.
  Args:
    directory: Directory where the data is located.
        If `labels` is "inferred", it should contain
        subdirectories, each containing images for a class.
        Otherwise, the directory structure is ignored.
    labels: Either "inferred"
        (labels are generated from the directory structure),
        None (no labels),
        or a list/tuple of integer labels of the same size as the number of
        image files found in the directory. Labels should be sorted according
        to the alphanumeric order of the image file paths
        (obtained via `os.walk(directory)` in Python).
    label_mode:
        - 'int': means that the labels are encoded as integers
            (e.g. for `sparse_categorical_crossentropy` loss).
        - 'categorical' means that the labels are
            encoded as a categorical vector
            (e.g. for `categorical_crossentropy` loss).
        - 'binary' means that the labels (there can be only 2)
            are encoded as `float32` scalars with values 0 or 1
            (e.g. for `binary_crossentropy`).
        - None (no labels).
    class_names: Only valid if "labels" is "inferred". This is the explicit
        list of class names (must match names of subdirectories). Used
        to control the order of the classes
        (otherwise alphanumerical order is used).
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
    crop_to_aspect_ratio: If True, resize the images without aspect
      ratio distortion. When the original aspect ratio differs from the target
      aspect ratio, the output image will be cropped so as to return the largest
      possible window in the image (of size `image_size`) that matches
      the target aspect ratio. By default (`crop_to_aspect_ratio=False`),
      aspect ratio may not be preserved.
    **kwargs: Legacy keyword arguments.
  Returns:
    A `tf.data.Dataset` object.
      - If `label_mode` is None, it yields `float32` tensors of shape
        `(batch_size, image_size[0], image_size[1], num_channels)`,
        encoding images (see below for rules regarding `num_channels`).
      - Otherwise, it yields a tuple `(images, labels)`, where `images`
        has shape `(batch_size, image_size[0], image_size[1], num_channels)`,
        and `labels` follows the format described below.
  Rules regarding labels format:
    - if `label_mode` is `int`, the labels are an `int32` tensor of shape
      `(batch_size,)`.
    - if `label_mode` is `binary`, the labels are a `float32` tensor of
      1s and 0s of shape `(batch_size, 1)`.
    - if `label_mode` is `categorial`, the labels are a `float32` tensor
      of shape `(batch_size, num_classes)`, representing a one-hot
      encoding of the class index.
  Rules regarding number of channels in the yielded images:
    - if `color_mode` is `grayscale`,
      there's 1 channel in the image tensors.
    - if `color_mode` is `rgb`,
      there are 3 channel in the image tensors.
    - if `color_mode` is `rgba`,
      there are 4 channel in the image tensors.
  """
  if 'smart_resize' in kwargs:
    crop_to_aspect_ratio = kwargs.pop('smart_resize')
  if kwargs:
    raise TypeError(f'Unknown keywords argument(s): {tuple(kwargs.keys())}')
  if labels not in ('inferred', None):
    if not isinstance(labels, (list, tuple)):
      raise ValueError(
          '`labels` argument should be a list/tuple of integer labels, of '
          'the same size as the number of image files in the target '
          'directory. If you wish to infer the labels from the subdirectory '
          'names in the target directory, pass `labels="inferred"`. '
          'If you wish to get a dataset that only contains images '
          f'(no labels), pass `labels=None`. Received: labels={labels}')
    if class_names:
      raise ValueError('You can only pass `class_names` if '
                       f'`labels="inferred"`. Received: labels={labels}, and '
                       f'class_names={class_names}')
  if label_mode not in {'int', 'categorical', 'binary', None}:
    raise ValueError(
        '`label_mode` argument must be one of "int", "categorical", "binary", '
        f'or None. Received: label_mode={label_mode}')
  if labels is None or label_mode is None:
    labels = None
    label_mode = None
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
  image_paths, labels, class_names = dataset_utils.index_directory(
      directory,
      labels,
      formats=ALLOWLIST_FORMATS,
      class_names=class_names,
      shuffle=shuffle,
      seed=seed,
      follow_links=follow_links)

  if label_mode == 'binary' and len(class_names) != 2:
    raise ValueError(
        f'When passing `label_mode="binary"`, there must be exactly 2 '
        f'class_names. Received: class_names={class_names}')

  image_paths, labels = dataset_utils.get_training_or_validation_split(
      image_paths, labels, validation_split, subset)
  if not image_paths:
    raise ValueError(f'No images found in directory {directory}. '
                     f'Allowed formats: {ALLOWLIST_FORMATS}')

  dataset = paths_and_labels_to_dataset(
      image_paths=image_paths,
      image_size=image_size,
      num_channels=num_channels,
      labels=labels,
      label_mode=label_mode,
      num_classes=len(class_names),
      interpolation=interpolation,
      crop_to_aspect_ratio=crop_to_aspect_ratio,
      stride=stride)
  if shuffle:
    # Shuffle locally at each iteration
    dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
  dataset = dataset.prefetch(tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True)
  # Users may need to reference `class_names`.
  dataset.class_names = class_names
  # Include file paths for images as attribute.
  dataset.file_paths = image_paths

  # Set shape for the batches (Added drop_remainder=True in dataset.batch() above to ensure all batches has the same size)
  dataset = dataset.map(lambda batch: set_shape(batch, batch_size=batch_size, image_size=image_size, num_channels=num_channels))
  return dataset


def paths_and_labels_to_dataset(image_paths,
                                image_size,
                                num_channels,
                                labels,
                                label_mode,
                                num_classes,
                                interpolation,
                                crop_to_aspect_ratio=False,
                                stride=None):
  """Constructs a dataset of images and labels."""
  # TODO(fchollet): consider making num_parallel_calls settable
  # return img_ds
  path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
  args = (image_size, num_channels, interpolation, crop_to_aspect_ratio)
  # Map path to sub-images
  img_ds = path_ds.map(lambda x: tf.py_function(load_image, [x, *args], [tf.float32]))
  # Flatten img_ds
  img_ds = img_ds.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
  return img_ds


def load_image(path, image_size, num_channels, interpolation,
               crop_to_aspect_ratio=False, stride=None):
  """
  Load an image from a path and return a list of sub images.
  Crop the images using smart_resize() if crop_to_aspect_ratio is True.
  Crop the images to sub images if not.
  Return:
    A list of sub images (length = 1 if using smart_resize).
  """
  img = tf.io.read_file(path)
  img = tf.image.decode_image(img, channels=num_channels, expand_animations=False)
  if crop_to_aspect_ratio:
    img = keras_image_ops.smart_resize(img, image_size,
                                       interpolation=interpolation)
    img.set_shape((image_size[0], image_size[1], num_channels))
    img = [img]
  else:
    # img = tf.image.resize(img, image_size, method=interpolation)
    img = crop_to_sub_images(img, image_size, num_channels, stride)
  return np.array(img)

def crop_to_sub_images(img, image_size, num_channels, stride=None):
    """
    Crop an image to sub images of image_size.
    Args:
        img: original image
        image_size: size of each sub image. (height, width)
        stride: A (height, width) tuple representing the number of pixels between sub images. If None then it is set to 2/3 of image_size to allow some overlappings.
    Return:
        A list of sub images.
    """
    if image_size[0] > img.shape[0] or image_size[1] > img.shape[1]:
        print("Error when cropping image: image_size > original image size.")
    upper_boundary = (img.shape[0] - image_size[0], img.shape[1] - image_size[1])
    if stride is None: 
        stride = (image_size[0] * 3 // 4, image_size[1] * 3 // 4)

    sub_images = []
    w = 0
    while True: # Loop until width reaches upper boundary
        if w > upper_boundary[1]:
            w = upper_boundary[1]
        h = 0
        while True: # Loop until height reaches upper boundary
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