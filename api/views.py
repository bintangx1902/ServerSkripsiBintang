import tempfile
from shutil import copyfileobj

from django.conf import settings
from rest_framework import status
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView

from pydub import AudioSegment
from api.serilalizers import FileUploadSerializer
from utils.loader import *

model_path = os.path.join(settings.STATIC_DIR, 'model', 'model_audio.h5')
audioModel = AudioModel(model_path)
model_path = os.path.join(settings.STATIC_DIR, 'model', 'model_image.h5')
imageModel = ImageModel(model_path)


class PredictVideo(APIView):
    def get(self, request, *args, **kwargs):
        return

    def post(self, request, *args, **kwargs):
        return


class PredictAudio(APIView):
    renderer_classes = [JSONRenderer, ]

    def get(self, request, *args, **kwargs):
        return Response(status=status.HTTP_405_METHOD_NOT_ALLOWED)

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
                data = load_audio_data(divided_audio)

                predictions = audioModel.predict(data)
                prediction_results, confidence_result = [], []
                for pred in predictions:
                    prediction = np.argmax(pred)
                    confidence = np.max(pred)
                    confidence = 100.0 if confidence * 100 > 1.0 else confidence * 100
                    prediction_results.append(
                        next((k for k, v in settings.AUDIO_DICT.items() if v == prediction), None))
                    confidence_result.append(confidence)

                return Response(data={'predictions': prediction_results, 'confidence': confidence_result},
                                status=status.HTTP_200_OK)

            except Exception as e:
                message = f"Error while predicting audio file: {e}"
                return Response({'error': message}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            finally:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

        return Response({'error': serializer.errors['file']}, status=status.HTTP_400_BAD_REQUEST)


class PredictImage(APIView):
    def get(self, request, *args, **kwargs):
        return Response({}, status=status.HTTP_405_METHOD_NOT_ALLOWED)

    def post(self, request, *args, **kwargs):
        serializer = FileUploadSerializer(data=request.data)
        if serializer.is_valid():
            file = self.request.FILES['file']
            temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=".jpg")

            for chunk in file.chunks():
                temp_file.write(chunk)
            temp_file.flush()
            temp_file.close()

            try:
                # save the file into temp
                file = load_image_data(file)
            except Exception as e:
                pass
            finally:
                pass

        return Response(data={
            'dir': dir(serializer),
        })
