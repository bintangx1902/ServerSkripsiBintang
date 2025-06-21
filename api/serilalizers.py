from rest_framework.serializers import Serializer, FileField


class FileUploadSerializer(Serializer):
    file = FileField()