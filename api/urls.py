from .views import *
from django.urls import path


urlpatterns = [
    path('audio', PredictAudio.as_view(), name='predict-audio'),
    path('image', PredictImage.as_view(), name='predict-image'),
    path('video', PredictVideo.as_view(), name='predict-video'),
]