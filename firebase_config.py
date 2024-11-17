# firebase_config.py
import firebase_admin
from firebase_admin import credentials, storage

def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate('chat.json')
        firebase_admin.initialize_app(cred, {
            'storageBucket': "chat-18938.appspot.com"
        })
