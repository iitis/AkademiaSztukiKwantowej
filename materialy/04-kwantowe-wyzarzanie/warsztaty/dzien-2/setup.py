import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def setup_paths():    
    sys.path.append(os.path.join(ROOT,
                    "pliki_pomocnicze"))


def setup():
    setup_paths()
