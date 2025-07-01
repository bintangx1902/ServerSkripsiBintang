import numpy as np
import pandas as pd
import tensorflow as tf
import moviepy.editor as mp
from os import path
import random
from PIL import Image


def plot_model(model: tf.keras.Model, show_shapes=False, show_layer_names=False):
    """
    :param model: the model you want to plot
    :param show_shapes: if you want to show the shape each layer of the model
    :param show_layer_names: if you want to show the layer names each layer of the model
    :return: 
    """
    return tf.keras.utils.plot_model(model, show_shapes=show_shapes, show_layer_names=show_layer_names)


def stratified_downsample(x_img, x_audio, y_img, y_audio):
    """
    Downsamples the image and audio datasets while ensuring each class is preserved
    and the number of samples matches the smallest class size in both datasets.
    It also ensures that x_img[i] corresponds to y_img[i] and x_audio[i] corresponds to y_audio[i].

    Args:
        x_img (np.array): Image inputs.
        x_audio (np.array): Audio inputs.
        y_img (np.array): Image labels.
        y_audio (np.array): Audio labels.

    Returns:
        Tuple of downsampled (x_img, x_audio, y_img, y_audio).
    """
    # Ensure seed for reproducibility
    np.random.seed(42)

    # Determine the minimum number of samples to match the smallest dataset size
    min_samples = min(len(x_img), len(x_audio))

    # Prepare lists to store downsampled indices
    selected_indices_img = []
    selected_indices_audio = []

    # Downsample image data to match the smallest class size in both datasets
    for cls in np.unique(y_img):
        # Find indices for this class in image data
        indices_img = np.where(y_img == cls)[0]
        # Shuffle and select the minimum number of samples for the class
        np.random.shuffle(indices_img)
        selected_indices_img.extend(indices_img[:min_samples])

    # Downsample audio data to match the smallest class size in both datasets
    for cls in np.unique(y_audio):
        # Find indices for this class in audio data
        indices_audio = np.where(y_audio == cls)[0]
        # Shuffle and select the minimum number of samples for the class
        np.random.shuffle(indices_audio)
        selected_indices_audio.extend(indices_audio[:min_samples])

    # Convert to numpy array for indexing
    selected_indices_img = np.array(selected_indices_img)
    selected_indices_audio = np.array(selected_indices_audio)

    # Sort the indices so that the data and labels remain consistent
    selected_indices_img = np.sort(selected_indices_img)
    selected_indices_audio = np.sort(selected_indices_audio)

    # Downsample both image and audio data using the selected indices
    x_img_ds = x_img[selected_indices_img]
    y_img_ds = y_img[selected_indices_img]
    x_audio_ds = x_audio[selected_indices_audio]
    y_audio_ds = y_audio[selected_indices_audio]

    return x_img_ds, x_audio_ds, y_img_ds, y_audio_ds


def downsample_df(df1, df2):
    target = min(len(df1), len(df2))
    new_df1 = balance_dataframe(df1, target)
    new_df2 = balance_dataframe(df2, target)
    return new_df1, new_df2


def balance_dataframe(df, target_total):
    labels = df['label'].unique()
    num_labels = len(labels)
    samples_per_label = target_total // num_labels

    balanced_parts = []
    for label in labels:
        df_label = df[df['label'] == label]
        sample_n = min(samples_per_label, len(df_label))
        balanced_parts.append(df_label.sample(n=sample_n, random_state=42))

    balanced_df = pd.concat(balanced_parts)

    # If we're short due to label imbalance, fill the rest randomly
    if len(balanced_df) < target_total:
        remaining = target_total - len(balanced_df)
        available_df = df.drop(balanced_df.index)
        if not available_df.empty:
            filler = available_df.sample(n=min(remaining, len(available_df)), random_state=42)
            balanced_df = pd.concat([balanced_df, filler])

    return balanced_df.reset_index(drop=True)


def create_temporary_file(input_file):
    return input_file


class VideoParser:
    """
    usage :
    parser = VideoParser("path/to/video.mp4")
    parser.get_video_without_audio("video_no_audio.mp4")
    parser.get_audio("audio.mp3")
    """

    def __init__(self, video_path, output_path=None):
        self.video_path = video_path
        self.video_clip = mp.VideoFileClip(video_path)
        self.output_path = output_path

    def split_video(self, output_path=None):
        video_no_audio = self.video_clip.without_audio()
        path_ = output_path if output_path is not None else self.output_path
        path_ = path.join(path_, 'vid_clean.mp4')
        video_no_audio.write_videofile(path_, codec="libx264", audio=False)
        return path_

    def split_audio(self, output_path=None):
        audio = self.video_clip.audio
        _path = output_path if output_path is not None else self.output_path
        _path = path.join(_path, 'audio_clean.mp3')
        audio.write_audiofile(_path)
        return _path

    def extract_image(self, output_path=None):
        output_path = output_path if output_path is not None else self.output_path

        duration = self.video_clip.duration
        interval = 3
        num_images = int(duration // interval) + (1 if duration % interval >= 1 else 0)
        image_paths = []

        for i in range(num_images):
            start = i * interval
            end = min((i * 1) + interval, duration)
            random_time = random.uniform(start, end)
            frame = self.video_clip.get_frame(random_time)
            image = Image.fromarray(np.uint8(frame))
            image_path = path.join(self.output_path, f"frame_{i + 1}.jpg")
            image.save(image_path)
            image_paths.append(image_path)

        return image_paths

    def get(self, output_path=None):
        output_path = output_path if output_path is not None else self.output_path
        return self.split_video(output_path), self.split_audio(output_path)
