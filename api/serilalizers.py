from rest_framework.serializers import Serializer, FileField
from rest_framework import serializers

class FileUploadSerializer(Serializer):
    file = FileField()


class VideoSerializer(Serializer):
    predicted_classes = serializers.ListField(child=serializers.CharField())
    confidences = serializers.ListField(child=serializers.FloatField())


class AudioSerializer(Serializer):
    predicted_classes = serializers.ListField(child=serializers.CharField())
    confidences = serializers.ListField(child=serializers.FloatField())


class OutputSerializer(Serializer):
    video = VideoSerializer()
    audio = AudioSerializer()
