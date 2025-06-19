import os
from pathlib import Path

from NurtrDjango.settings import *

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',    # Tells Django to use PostgreSQL
        'NAME': 'nurtr_django',                       # Database name (actual app database)
        'USER': 'postgres',                           # Username we just saw
        'PASSWORD': 'NurtrSecure2024!',
        'HOST': '127.0.0.1',                         # Local host (through proxy)
        'PORT': '5433',                               # Port where proxy is listening
    }
}

