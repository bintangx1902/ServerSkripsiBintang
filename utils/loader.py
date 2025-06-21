import os

import tensorflow as tf
from audio import preprocess_audio



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

def load_data(file, model):
    spectrogram = preprocess_audio(file)
    # spectrogram = add_noise(spectrogram)
    return spectrogram
