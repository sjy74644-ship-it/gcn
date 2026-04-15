"""Visual-to-radar SRRL pose distillation framework (single-frame)."""

from .dataset import DatasetSpec, ImageOnlyDataset, PairedPosePngDataset
from .models import PoseStudent, PoseTeacher, SRRLProjector
from .trainer import DistillationConfig, DistillationTrainer

__all__ = [
    "DatasetSpec",
    "PairedPosePngDataset",
    "ImageOnlyDataset",
    "PoseTeacher",
    "PoseStudent",
    "SRRLProjector",
    "DistillationConfig",
    "DistillationTrainer",
]
