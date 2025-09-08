import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def setup():    
    sys.path.append(os.path.join(ROOT, "pliki_pomocnicze"))

print(os.path.join(ROOT, "pliki_pomocnicze"))