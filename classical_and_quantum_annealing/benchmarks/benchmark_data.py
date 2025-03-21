import os
import sys
from collections import namedtuple
from dataclasses import dataclass

fp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if fp_path not in sys.path:
    sys.path.insert(0, fp_path)

from funkcje_pomocnicze import read_instance


@dataclass
class Instance:
    path: os.PathLike
    best_energy: float
    name: str

    def __init__(self, path: os.PathLike, best_energy: float, name: str):
        self.path = path
        self.best_energy = best_energy
        self.name = name

instance = namedtuple("instance", ["path", "best_energy", "name"])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

P2 = instance(os.path.join(ROOT, "instancje", "Pegasus", "P2_CBFM-P.txt"), -39.0, "P2")
P4 = instance(os.path.join(ROOT, "instancje", "Pegasus", "P4_CBFM-P.txt"), -469.0, "P4")
P8 = instance(os.path.join(ROOT, "instancje", "Pegasus", "P8_CBFM-P.txt"), -2752.0, "P8")
P12 = instance(os.path.join(ROOT, "instancje", "Pegasus", "P12_CBFM-P.txt"), -6831.0, "P12")
P16 = instance(os.path.join(ROOT, "instancje", "Pegasus", "P16_CBFM-P.txt"), -12772.0, "P16")

C2 = instance(os.path.join(ROOT, "instancje", "Chimera", "C2_random.txt"), -37.25, "C2")
C4 = instance(os.path.join(ROOT, "instancje", "Chimera", "C4_random.txt"), -150.0, "C4")
C8 = instance(os.path.join(ROOT, "instancje", "Chimera", "C8_random.txt"), -642.5, "C8")
C12 = instance(os.path.join(ROOT, "instancje", "Chimera", "C12_random.txt"), -1417.75, "C12")
C16 = instance(os.path.join(ROOT, "instancje", "Chimera", "C16_random.txt"), -2551.0, "C16")

Z2 = instance(os.path.join(ROOT, "instancje", "Zephyr", "Z2_random.txt"), -310.75, "Z2")
Z4 = instance(os.path.join(ROOT, "instancje", "Zephyr", "Z4_random.txt"), -1184.75, "Z4")
Z8 = instance(os.path.join(ROOT, "instancje", "Zephyr", "Z8_random.txt"), -4697.5, "Z8")
Z12 = instance(os.path.join(ROOT, "instancje", "Zephyr", "Z12_random.txt"), -10289.0, "Z12")
Z15 = instance(os.path.join(ROOT, "instancje", "Zephyr", "Z15_random.txt"), -16087.5, "Z15") 

pegasus_data = [P2, P4, P8, P12, P16]
chimera_data = [C2, C4, C8, C12, C16]
zephyr_data = [Z2, Z4, Z8, Z12, Z15]
