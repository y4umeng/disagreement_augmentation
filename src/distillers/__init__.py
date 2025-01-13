from ._base import Vanilla
from .KD import KD
from .DA import DA
from .MLKD import MLKD

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "MLKD": MLKD
}
