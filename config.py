import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Pytorch + learning
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 24
NUM_WORKERS = 2
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500

# File Location
FILE_PREFIX = "Anime"
CHECKPOINT_DISC = f"Data/{FILE_PREFIX}/SavedModel/dis.pth.tar"
CHECKPOINT_GEN = f"Data/{FILE_PREFIX}/SavedModel/gen.pth.tar"
TRAIN_DIR = f"Data/{FILE_PREFIX}/TrainingData"
VAL_DIR = f"Data/{FILE_PREFIX}/TestingData"
EVAL_DIR  = f"Data/{FILE_PREFIX}/Evaluation"
PEEP_DIR  = f"Data/{FILE_PREFIX}/SeeAll"

# Image data
IMAGE_HALF_X = 4624 if FILE_PREFIX.find('Anime') == -1 else 512  # 512
IMAGE_SIZE_X = 256  # 2312  # 1024
IMAGE_SIZE_Y = 256  # 1734

# Manual Options
LOAD_MODEL = True
SAVE_MODEL = False
OBSERVE_MODEL = True


# Helper Methods
bothTransform = A.Compose(
    [A.Resize(width=IMAGE_SIZE_X, height=IMAGE_SIZE_Y), ], additional_targets={"image0": "image"},
)

transformOnlyInput = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
        ToTensorV2(),
    ]
)

transformOnlyOutput = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
        ToTensorV2(),
    ]
)


# Creates all necessary folders
if __name__ == '__main__':
    from pathlib import Path
    for p in [TRAIN_DIR, VAL_DIR, EVAL_DIR, PEEP_DIR]:
        Path(p).mkdir(parents=True, exist_ok=True)
    upTo = len(f"Data/{FILE_PREFIX}/SavedModel/")
    for p in [CHECKPOINT_GEN, CHECKPOINT_DISC]:
        Path(p[:upTo]).mkdir(parents=True, exist_ok=True)

