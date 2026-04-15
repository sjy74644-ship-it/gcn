"""Visual-to-radar SRRL pose distillation framework."""

from .dataset import DatasetSpec, PairedPosePngDataset
from .models import PoseTeacher, PoseStudent, SRRLProjector
from .trainer import DistillationConfig, DistillationTrainer

__all__ = [
    "DatasetSpec",
    "PairedPosePngDataset",
    "PoseTeacher",
    "PoseStudent",
    "SRRLProjector",
    "DistillationConfig",
    "DistillationTrainer",
]
