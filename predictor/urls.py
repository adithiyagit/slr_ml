# urls.py for Calories Predictor
from django.urls import path
from .views import index, predict_calories

urlpatterns = [
    path('', index, name='index'),
    path('predict/', predict_calories, name='predict_calories'),
]
