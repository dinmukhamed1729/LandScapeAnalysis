from django.urls import path
from . import views

urlpatterns = [
    path('analyze/', views.AnalyzeView.as_view(), name='analyze'),
    path('status/<str:task_id>/', views.StatusView.as_view(), name='status'),
]
