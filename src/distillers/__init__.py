from ._base import Vanilla
from .KD import KD
from .DA import DA

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "DA": DA,
}
