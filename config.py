import os
import string
import random
basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    SECRET_KEY = '1234'
    SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL'] 
    key = ''.join( random.choice(string.ascii_letters + string.digits + string.ascii_uppercase) for i in range(12) )

class ProductionConfig(Config):
    DEBUG = False


class StagingConfig(Config):
    DEVELOPMENT = True
    DEBUG = True


class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True


class TestingConfig(Config):
    TESTING = True