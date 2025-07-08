import tempfile
from shutil import rmtree

from django.conf import settings
from rest_framework import status
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView

from api.serilalizers import FileUploadSerializer, OutputSerializer
from utils.loader import *
from utils.parser import *

model_path = os.path.join(settings.STATIC_DIR, 'model', 'model_audio.h5')
audioModel = AudioModel(model_path)
model_path = os.path.join(settings.STATIC_DIR, 'model', 'model_image.h5')
imageModel = ImageModel(model_path)


class PredictVideo(APIView):
    def get(self, request, *args, **kwargs):
        return Response(status=status.HTTP_403_FORBIDDEN)

    def post(self, request, *args, **kwargs):
        serializer = FileUploadSerializer(data=request.data)
        if serializer.is_valid():
            video = request.FILES.get('file', None)
            temp_file = tempfile.NamedTemporaryFile(delete=False, dir=settings.TEMPORARY_FILE_DIR, suffix='.mp4')
            for chunk in video.chunks():
                temp_file.write(chunk)
            temp_file.flush()
            temp_file.close()
            temp_filepath = ""
            try:
                # create new folder for output
                get_name_ext = str(temp_file.name).split('.')
                temp_filepath, filename_ext = get_name_ext[0], get_name_ext[-1]
                os.makedirs(os.path.dirname(temp_filepath), exist_ok=True)
                vid = VideoParser(temp_file.name)
                image_frames, audio = vid.get(temp_filepath)
                audio = split_audio(audio, temp_filepath)

                image_data = load_image_data(image_frames)
                audio_data = load_audio_data(audio)

                audio_predictions = audioModel.predict_data(audio_data)
                image_predictions = imageModel.predict_data(image_data)

                image_predictions, image_confidences = get_results(image_predictions, settings.IMAGE_DICT)
                audio_predictions, audio_confidences = get_results(audio_predictions, settings.AUDIO_DICT)

                output = {
                    "video": {
                        "predicted_classes": image_predictions,
                        "confidences": image_confidences
                    },
                    "audio": {
                        "predicted_classes": audio_predictions,
                        "confidences": audio_confidences
                    }
                }
                response = OutputSerializer(data=output)
                if response.is_valid():
                    return Response(response.data, status=status.HTTP_200_OK)
                return Response(response.errors, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            finally:
                if os.path.exists(temp_file.name):
                    os.remove(temp_file.name)
                    rmtree(temp_filepath)
        return None


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

                predictions = audioModel.predict_data(data)
                prediction_results, confidence_result = get_results(predictions, settings.AUDIO_DICT)

                return Response(data={'predictions': prediction_results, 'confidence': confidence_result},
                                status=status.HTTP_200_OK)

            except Exception as e:
                message = f"Error while predicting audio file: {e}"
                return Response({'error': message}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            finally:
                if os.path.exists(temp_path):
                    rmtree(temp_path)

        return Response({'error': serializer.errors['file']}, status=status.HTTP_400_BAD_REQUEST)


class PredictImage(APIView):
    def get(self, request, *args, **kwargs):
        return Response({}, status=status.HTTP_405_METHOD_NOT_ALLOWED)

    def post(self, request, *args, **kwargs):
        serializer = FileUploadSerializer(data=request.data)
        temp_dir = settings.TEMPORARY_FILE_DIR
        if serializer.is_valid():
            file = self.request.FILES['file']
            temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=".jpg")

            for chunk in file.chunks():
                temp_file.write(chunk)
            temp_file.flush()
            temp_file.close()

            try:
                data = load_image_data([temp_file.name])
                predictions = imageModel.predict_data(data)
                prediction_results, confidence_result = get_results(predictions, settings.IMAGE_DICT)

                return Response(data={'predictions': prediction_results, 'confidence': confidence_result},
                                status=status.HTTP_200_OK)

            except Exception as e:
                print(f"Error while predicting image file: {e}")
            finally:
                if os.path.exists(temp_file.name):
                    os.remove(temp_file.name)

        return Response({'error': serializer.errors['file']}, status=status.HTTP_400_BAD_REQUEST)
