import tempfile
from shutil import copyfileobj

import numpy as np
from django.conf import settings
from rest_framework import status
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView

from api.serilalizers import FileUploadSerializer
from utils.loader import *

model_path = os.path.join(settings.STATIC_PATH, 'model', 'model_audio.h5')
audioModel = AudioModel(model_path)
model_path = os.path.join(settings.STATIC_PATH, 'model', 'model_image.h5')
imageModel = ImageModel(model_path)

data_dict = {3: 'lapar', 1: 'bersendawa atau sendawa', 2: 'tidak nyaman', 0: 'sakit perut', 4: 'lelah atau letih'}


class PredictVideo(APIView):
    def get(self, request, *args, **kwargs):
        return

    def post(self, request, *args, **kwargs):
        return


class PredictAudio(APIView):
    renderer_classes = [JSONRenderer, ]

    def get(self, request, *args, **kwargs):
        return

    def post(self, request, *args, **kwargs):
        serializer = FileUploadSerializer(data=request.data)
        if serializer.is_valid():
            uploaded_file = self.request.FILES['file']
            ext_name: str = uploaded_file.name.split('.')[-1]

            if ext_name in settings.ALLOWED_AUDIO_EXTENSIONS:
                audio = AudioSegment.from_file(uploaded_file)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                    temp_path = os.path.join(settings.TEMPORARY_FILE_DIR, temp_file.name)
                    audio.export(temp_path, format='wav')
                    temp_filename = temp_path
            else:
                msg_list = [f".{x}" for x in settings.ALLOWED_AUDIO_EXTENSIONS]
                msg = ' or '.join(msg_list)
                return Response({'error': f"File format extension not supported! Only {msg} are supported!"},
                                status=status.HTTP_403_FORBIDDEN)

            try:
                get_name_ext = temp_filename.split('.')
                temp_filepath, filename_ext = get_name_ext[0], get_name_ext[-1]
                os.makedirs(temp_filepath, exist_ok=True)
                
                divided_audio = split_audio(temp_filename, temp_filepath)
                data = load_data(divided_audio)

                pred = self.model.predict(data)
                prediction = np.argmax(pred)
                confidence = np.max(pred)
                confidence = 100.0 if confidence * 100 > 1.0 else confidence
                pred_result = data_dict[prediction]
            except Exception as e:
                message = f"Error while predicting audio file: {e}"
                return Response({'error': message}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            finally:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

            # hiraukan karena data yang masuk ga akan di save
            save_dir = os.path.join(settings.MEDIA_ROOT, 'result', str(prediction))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, uploaded_file.name)
            with open(save_path, 'wb+') as destination:
                copyfileobj(uploaded_file, destination)

            return Response(data={'predictions': pred_result, 'confidence': confidence}, status=status.HTTP_200_OK)

        return Response({'error': serializer.errors['file']}, status=status.HTTP_400_BAD_REQUEST)


class PredictImage(APIView):
    def get(self, request, *args, **kwargs):
        return

    def post(self, request, *args, **kwargs):
        return
