from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-insecure-landscape-analysis-secret-key-change-in-prod'

DEBUG = True

ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.contenttypes',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'api',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
]

ROOT_URLCONF = 'core.urls'

WSGI_APPLICATION = 'core.wsgi.application'

DATABASES = {}

STATIC_URL = '/static/'

CORS_ALLOW_ALL_ORIGINS = True

MEDIA_ROOT = BASE_DIR / 'media'
MEDIA_URL = '/media/'

DATA_UPLOAD_MAX_MEMORY_SIZE = 200 * 1024 * 1024  # 200 MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 200 * 1024 * 1024

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
