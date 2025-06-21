import random

import cv2
import numpy as np
import tensorflow as tf


def preprocess_image(img_path, dsize: tuple = (128, 128), normalize=False, augment=False,
                     rescale: float = None, preprocess_function=None):
    """
    Preprocesses an image with optional normalization, rescaling, augmentation, or custom preprocessing.

    Args:
        img_path (str): Path to the image (relative or absolute).
        dsize (tuple): Target size for resizing (width, height), default is (128, 128).
        normalize (bool): If True, normalize using ImageNet mean and std.
        augment (bool): If True, apply one random augmentation (flip, rotate, or zoom).
        rescale (float, optional): If provided, rescale image pixel values by this factor.
        preprocess_function (callable, optional): Custom preprocessing function, e.g., tf.keras.applications.vgg19.preprocess_input.

    Returns:
        np.ndarray or tf.Tensor: Processed image as a NumPy array or TensorFlow tensor (if preprocess_function is used).

    Raises:
        FileNotFoundError: If the image path is invalid.
        ValueError: If input parameters are invalid (e.g., negative dsize, conflicting options).
    """
    if preprocess_function is not None and not callable(preprocess_function):
        raise ValueError("preprocess_function must be a callable function.")

    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize, interpolation=cv2.INTER_LANCZOS4)

    if normalize and rescale is None:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

    if rescale is not None and not normalize:
        image = np.array(image) * rescale

    if augment:
        # Choose one augmentation randomly
        augmentations = ['flip_horizontal', 'flip_vertical', 'rotate', ]
        chosen_aug = random.choice(augmentations)

        if chosen_aug == 'flip_horizontal':
            image = cv2.flip(image, 1)
        elif chosen_aug == 'flip_vertical':
            image = cv2.flip(image, 0)
        elif chosen_aug == 'rotate':
            # Random rotation between -30 and 30 degrees
            angle = random.uniform(-30, 30)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        elif chosen_aug == 'zoom':
            # Random zoom between 0.8x and 1.2x
            zoom_factor = random.uniform(0.8, 1.2)
            h, w = image.shape[:2]
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            if zoom_factor > 1.0:
                # Zoom in: crop the center
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                start_x = (new_w - w) // 2
                start_y = (new_h - h) // 2
                image = resized[start_y:start_y + h, start_x:start_x + w]
            else:
                # Zoom out: pad with replicated borders
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                pad_x = (w - new_w) // 2
                pad_y = (h - new_h) // 2
                image = cv2.copyMakeBorder(resized, pad_y, pad_y, pad_x, pad_x,
                                           cv2.BORDER_REPLICATE)

    if preprocess_function is not None:
        return np.array(preprocess_function(image))
    return image


def preprocess_image_tf(img_path, dsize: tuple = (128, 128), decode=False, cv=False, normalize=False, augment=False,
                        rescale: float = None):
    if decode:
        image = tf.io.read_file(img_path)
        image = tf.io.decode_image(image, expand_animations=False, dtype=tf.float32, channels=3)
    else:
        image = tf.keras.utils.load_img(img_path)
        image = tf.keras.preprocessing.image.img_to_array(image)
        if image.shape[-1] == 1:  # grayscale
            image = tf.image.grayscale_to_rgb(image)
        image = tf.convert_to_tensor(image, dtype=tf.float32)  # Ensure it's a tensor

    if cv:
        image_np = image.numpy() if isinstance(image, tf.Tensor) else image
        image_np = cv2.resize(image_np, dsize, interpolation=cv2.INTER_LANCZOS4)
        image = tf.convert_to_tensor(image_np, dtype=tf.float32)
    else:
        image = tf.image.resize(image, dsize, method='lanczos5')

    if normalize:
        image = image / 255.0

    if rescale is not None and not normalize:
        image = image * rescale

    return image
