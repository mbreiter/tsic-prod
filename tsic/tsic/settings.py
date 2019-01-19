"""
settings for tsic.

MATTHEW REITER

"""

import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ['SECRET_KEY_1'] + '#f' + os.environ['SECRET_KEY_2']

CURRENCY_KEY = os.environ['CURRENCY_KEY']
APLHA_VANTAGE_KEY = os.environ['APLHA_VANTAGE_KEY']
QUANDL_KEY = os.environ['QUANDL_KEY']
INTRIO_ID = os.environ['INTRIO_ID']
INTRIO_PASS = os.environ['INTRIO_PASS']

USER_TYPES =  ('Staff', 'Analyst', 'Associate', 'Quant')
USER_TYPE_KEYS = (os.environ['STAFF_PASS'], os.environ['ANALYST_PASS'], os.environ['ASSOCIATE_PASS'], os.environ['QUANT_PASS'])

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True
ALLOWED_HOSTS=['localhost', '178.128.230.231']

# Application definition
LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/'

# Celery application definition
BROKER_URL = 'redis://localhost:6379'
CELERY_RESULT_BACKEND = 'redis://localhost:6379'
CELERY_ACCEPT_CONTENT = ['application/json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIME_ZONE = 'America/Toronto'
CELERY_SEND_TASK_SENT_EVENT = True

# Django email stuff
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_USE_TLS = True
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_HOST_USER = 'tsic.quant@gmail.com'
EMAIL_HOST_PASSWORD = os.environ['EMAIL_HOST_PASSWORD']
EMAIL_PORT = 587

INSTALLED_APPS = [
    'website',
    'funds',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'tsic.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, '/website/templates'), os.path.join(BASE_DIR, '/website/static'),],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

AUTH_USER_MODEL = 'website.User'

WSGI_APPLICATION = 'tsic.wsgi.application'

# Database
# https://docs.djangoproject.com/en/2.0/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'tsic_prod',
        'USER': 'mreiter',
        'PASSWORD': os.environ["DB_PROD_PASSWORD"],
        'HOST': 'localhost',
        'PORT': '',
    }
}

# Password validation
# https://docs.djangoproject.com/en/2.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

ADMINS = (
    ('Matthew Reiter', 'matthew.reiter@mail.utoronto.ca'),
)

MANAGERS = (
    ('Matthew Reiter', 'matthew.reiter@mail.utoronto.ca'),
)

SESSION_EXPIRE_AT_BROWSER_CLOSE = True

# Internationalization
# https://docs.djangoproject.com/en/2.0/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'America/Toronto'

USE_TZ = False

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.0/howto/static-files/

STATIC_ROOT = os.path.join(BASE_DIR, 'static/')
STATIC_URL = '/static/'

MEDIA_URL = '/documents/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'website/static/documents')
