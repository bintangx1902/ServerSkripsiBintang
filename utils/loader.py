import os

import numpy as np
import pandas as pd
import tensorflow as tf
from pydub import AudioSegment

from .audio import preprocess_audio
from .image import preprocess_image


class AudioModel:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def create_model(self):
        base_model = tf.keras.applications.VGG19(
            weights='imagenet',
            include_top=False,
            input_tensor=tf.keras.layers.Input(shape=(128, 110, 3)),
            pooling="avg",
        )

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(8, activation='softmax'),
        ])
        return model

    def load_model(self, model_path):
        model = self.create_model()
        if os.path.exists(model_path):
            try:
                model.load_weights(model_path)
                print("Model loaded successfully from", model_path)
                model.summary()
            except Exception as e:
                print("Error loading model:", e)
            finally:
                print('Finished loading audio model')
        else:
            print("Model file not found at:", model_path)
        return model

    def predict(self, audio_spectrogram):
        return self.model.predict(audio_spectrogram)


class ImageModel:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def create_model(self):
        base = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet',
            input_shape=(100, 100, 3),
            pooling=None
        )

        input_image = tf.keras.Input(shape=(100, 100, 3), name='input_image')

        x = base(input_image)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        out1 = tf.keras.layers.Dense(7, activation='softmax', name='image_class')(x)

        model = tf.keras.models.Model(inputs=input_image, outputs=out1)
        return model

    def load_model(self, model_path):
        model = self.create_model()
        if os.path.exists(model_path):
            try:
                model.load_weights(model_path)
                print("Model loaded successfully from", model_path)
                model.summary()
            except Exception as e:
                print("Error loading model:", e)
            finally:
                print('Finished loading image model')
        else:
            print("Model file not found at:", model_path)

        return model


def load_audio_data(file_dir_list):
    audio = pd.DataFrame({'filepath': file_dir_list, })
    with tf.device('/GPU:0'):
        audio['data'] = audio.filepath.apply(preprocess_audio)

    x_audio_train = np.stack(audio['data'].values)
    return x_audio_train


def load_image_data(file_dir_list):
    image = pd.DataFrame({'filepath': file_dir_list, })
    with tf.device('/GPU:0'):
        image['data'] = image['filepath'].apply(
            lambda x: preprocess_image(x, (128, 110),
                                       preprocess_function=tf.keras.applications.vgg19.preprocess_input,
                                       augment=True)
        )
    return np.stack(image['data'].values)


def split_audio(input_file, output_dir, segment_duration=3000) -> list:
    """
    Split audio file into segments of specified duration (in milliseconds).

    Args:
        input_file: Path to input audio file
        output_dir: Directory to save output segments
        segment_duration: Duration of each segment in milliseconds (default: 3000ms = 3 seconds)

    Returns:
        List of paths to saved segment files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize list to store output paths
    saved_paths = []

    # Load audio file
    audio = AudioSegment.from_file(input_file, format="wav" if hasattr(input_file, 'read') else None)

    # Get total duration in milliseconds
    total_duration = len(audio)

    # Calculate number of full segments and remaining duration
    num_segments = total_duration // segment_duration
    remainder = total_duration % segment_duration

    # Split and export segments
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = start_time + segment_duration
        segment = audio[start_time:end_time]

        # Generate output filename
        output_file = os.path.join(output_dir, f"segment_{i + 1:03d}.wav")
        segment.export(output_file, format="wav")
        saved_paths.append(output_file)

    # Export remainder if it exists
    if remainder > 0:
        start_time = num_segments * segment_duration
        segment = audio[start_time:]
        output_file = os.path.join(output_dir, f"segment_{num_segments + 1:03d}_remainder.wav")
        segment.export(output_file, format="wav")
        saved_paths.append(output_file)

    return saved_paths
