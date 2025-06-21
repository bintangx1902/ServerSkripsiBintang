#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

import gdown


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'detection.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


def check_and_download_model():
    print("*** Checking Model")
    url_audio = r'https://drive.google.com/file/d/171cFnI700UAWvGoBCYpKsnJmicZSE7ze/view?usp=sharing'
    url_image = r'https://drive.google.com/file/d/1bUc1adATx94d8I-kvdsMMjCM3BfdLjKz/view?usp=sharing'
    model_path = 'static/model'
    os.makedirs(model_path, exist_ok=True)

    print("*** Check Audio Model")
    if not os.path.exists(os.path.join(model_path, 'model_audio.h5')):
        print("*** Download Audio Model")
        gdown.download(url_audio, os.path.join(model_path, 'model_audio.h5'), fuzzy=True)

    print("*** Check Image Model")
    if not os.path.exists(os.path.join(model_path, 'model_image.h5')):
        print("*** Download Image Model")
        gdown.download(url_image, os.path.join(model_path, 'model_image.h5'), fuzzy=True)


if __name__ == '__main__':
    check_and_download_model()
    main()
