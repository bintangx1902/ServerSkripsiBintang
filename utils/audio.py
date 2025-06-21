import librosa
import numpy as np


def load_audio(file_content, target_sr):
    audio, sample_rate = librosa.load(file_content, sr=target_sr, mono=True)
    trim, _ = librosa.effects.trim(audio, top_db=20)
    wav_resampled = librosa.resample(trim, orig_sr=sample_rate, target_sr=target_sr)
    return wav_resampled, target_sr


def preprocess_audio(file_content, target_sr=32000, n_mels=128, fmin=80.0, fmax=7600.0, fixed_frames=110):
    """
    :param file_content: file path (full path or can be absolute path)  
    :param target_sr: if you have your own desired sampling rate
    :param n_mels: max length of mel-spectrogram
    :param fmin: minimum frequency
    :param fmax: maximum frequency
    :return: the mel-spectrogram cast as image with 3 channels
    """
    audio, sample_rate = load_audio(file_content, target_sr)
    # target_samples = int(np.ceil(len(audio) / sample_rate) * target_sr)
    # if len(audio) < target_samples:
    #     audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=target_sr,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.min)

    if mel_spectrogram_db.shape[1] < fixed_frames:
        pad_width = fixed_frames - mel_spectrogram_db.shape[1]
        mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spectrogram_db = mel_spectrogram_db[:, :fixed_frames]

    spectogram = np.expand_dims(mel_spectrogram_db, axis=-1)
    return np.repeat(spectogram, 3, axis=-1)
