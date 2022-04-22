import shutil
import pandas as pd
from config import DATASET_PATH, LOGODET_3K_NORMAL_PATH
from pathlib import Path
from config import SEED
import numpy as np

import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

sys.path.append(str(ROOT / 'pycil'))
from utils.data_manager import DataManager


DETECTION4CIL_PATH = 'LogoDet-3K_det4cil'

args = {
    "dataset": "LogoDet-3K_cropped",
    "shuffle": True,
    "init_cls": 30,
    "increment": 10,
    "data_augmentation": True,
    "seed": SEED,
}

# Create directory
os.makedirs(Path(DATASET_PATH) / DETECTION4CIL_PATH, exist_ok=True)

# Copy images
if not (Path(DATASET_PATH) / DETECTION4CIL_PATH / 'images').exists():
    print(f'Copying images. . .')
    shutil.copytree(
        src=Path(DATASET_PATH) / LOGODET_3K_NORMAL_PATH / "images",
        dst=Path(DATASET_PATH) / DETECTION4CIL_PATH / "images"
    )

# Copy files
for file in ['train.txt', 'validation.txt', 'test.txt']:
    if not (Path(DATASET_PATH) / DETECTION4CIL_PATH / file).exists():
        print(f'Copying {file}. . .')
        shutil.copytree(
            src=Path(DATASET_PATH) / LOGODET_3K_NORMAL_PATH / file,
            dst=Path(DATASET_PATH) / DETECTION4CIL_PATH / file
        )

df_full = pd.read_pickle(Path(DATASET_PATH) / LOGODET_3K_NORMAL_PATH / 'metadata_full_images.pickle')
df_cropped = pd.read_pickle(Path(DATASET_PATH) / LOGODET_3K_NORMAL_PATH / 'metadata_cropped_images.pickle')

# Create dataset manager
data_manager = DataManager(
    args['dataset'], args['shuffle'], args['seed'], args['init_cls'],
    args['increment'], data_augmentation=args['data_augmentation']
)

train = data_manager.get_dataset(indices=np.arange(0, args['init_cls']), source='train', mode='train'),
validation = data_manager.get_dataset(indices=np.arange(0, args['init_cls']), source='validation', mode='test'),
test = data_manager.get_dataset(indices=np.arange(0, args['init_cls']), source='test', mode='test'),

# TODO: Generate labels
