"""Visual-to-radar pose distillation framework."""

from .dataset import PairedPosePngDataset
from .models import PoseTeacher, PoseStudent
from .trainer import DistillationTrainer, DistillationConfig

__all__ = [
    "PairedPosePngDataset",
    "PoseTeacher",
    "PoseStudent",
    "DistillationTrainer",
    "DistillationConfig",
]
