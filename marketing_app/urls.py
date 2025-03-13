from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("chatbot/", views.chatbot_view, name="chatbot"),
    path("get_kpi/", views.get_kpi, name="get_kpi"),  # Ajoutez cette ligne
]
