from django.contrib import admin
from django.urls import path, re_path, include
from django.conf import settings
from django.views.static import serve

from PneumoniaApp import views

urlpatterns = [
    path("", views.home.as_view()),
    path("admin/", admin.site.urls),
    re_path(r'^pneumonia/?$', views.call_model.as_view()),
    re_path(r'^pneumonia_url/?$', views.call_model_link.as_view()),
    re_path(r'^uploads/(?P<path>.*)$', serve,{'document_root': settings.UPLOADS_ROOT}),
]
