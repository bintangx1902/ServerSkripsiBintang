import os
import tempfile

from django.conf import settings
from pydub import AudioSegment
from rest_framework.response import Response
from rest_framework.views import APIView
from shutil import copyfileobj

from api.serilalizers import FileUploadSerializer
from utils.loader import AudioModel

model_path = os.path.join(settings.STATIC_PATH, 'model', 'model_audio.h5')
audioModel = AudioModel(model_path)
model_path = os.path.join(settings.STATIC_PATH, 'model', 'model_image.h5')
imageModel = AudioModel(model_path)

allowed_file_extension = ['mp3', 'wav', 'mp4', 'm4a']


class PredictVideo(APIView):
    def get(self, request, *args, **kwargs):
        return

    def post(self, request, *args, **kwargs):
        return


class PredictAudio(APIView):
    def get(self, request, *args, **kwargs):
        return

    def post(self, request, *args, **kwargs):
        serializer = FileUploadSerializer(data=request.data)
        if serializer.is_valid():
            uploaded_file = self.request.FILES['file']
            ext_name: str = uploaded_file.name.split('.')[-1]
        uploaded_file = self.request.FILES['file']
        ext_name: str = uploaded_file.name.split('.')[-1]

        if ext_name in self.allowed_file_extension:
            audio = AudioSegment.from_file(uploaded_file)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                audio.export(temp_file.name, format='wav')
                temp_filename = temp_file.name
        else:
            msg_list = [f".{x}" for x in self.allowed_file_extension]
            msg = ' or '.join(msg_list)
            return Response({'error': f"File format extension not supported! Only {msg} are supported!"},
                            status=status.HTTP_403_FORBIDDEN)

        try:
            data = load_data(temp_filename, self.inceptionv3)

            pred = self.model.predict(data)
            prediction = np.argmax(pred)
            confidence = np.max(pred)
            confidence = 100.0 if confidence * 100 > 1.0 else confidence
            pred_result = self.data_dict[prediction]

        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        return


class PredictImage(APIView):
    def get(self, request, *args, **kwargs):
        return

    def post(self, request, *args, **kwargs):
        return
