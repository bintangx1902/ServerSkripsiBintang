from django.urls import path
from .views import *


urlpatterns = [
    path('audio', PredictAudio.as_view(), name='predict-audio'),
    path('image', PredictImage.as_view(), name='predict-image'),
]