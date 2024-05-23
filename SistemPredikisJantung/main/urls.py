"""
URL configuration for main project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
    
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from . import views


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('about/', views.about, name='about'),
    path('form/', views.form, name='form'),
    # route
    path('upload/', views.upload_dataset, name='upload-dataset'),
    path('list/', views.list_dataset, name='list-dataset'),
    path('delete/<int:pk>/', views.delete_dataset, name='delete-dataset'),
    path('display/<int:pk>/', views.display_dataset, name='display-dataset'),
    path('result/<int:pk>/', views.create_model, name='create-model'),
    path('model/<int:pk>/', views.model, name='model'),
    path('predict-model/', views.predict_view, name='predict-model'),
    path('predict/', views.predict_view, name='predict'),
    path('form/hasil/', views.predict_view, name='predict'),
] + static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
