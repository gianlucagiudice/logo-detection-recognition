import shutil
import pandas as pd
from tqdm import tqdm

from config import DATASET_PATH, LOGODET_3K_NORMAL_PATH
from pathlib import Path
from config import SEED
import numpy as np
import argparse

import subprocess

import sys
import os

import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

sys.path.append(str(ROOT / 'pycil'))
from utils.data_manager import DataManager


DETECTION4CIL_PATH = 'LogoDet-3K_det4cil'


parser = argparse.ArgumentParser(description='Train yolo detector for CIL.')

parser.add_argument('--num-class', type=int, required=True,
                    help='Number of classes to use for training.')

parser.add_argument('--only-det', type=bool, required=False, default=False, action=argparse.BooleanOptionalAction,
                    help='Number of classes to use for training.')

parser.add_argument('--start-training', type=bool, required=True, default=False, action=argparse.BooleanOptionalAction,
                    help='Start the training of the model.')

cmd_args = parser.parse_args()

args = {
    "dataset": "LogoDet-3K_cropped",
    "shuffle": True,
    "init_cls": cmd_args.num_class,
    "increment": 10,
    "data_augmentation": True,
    "seed": SEED,

    # Training
    "img_size": 512,
    'batch_size': 32,
    "epochs": 30,
    "weights": 'yolov5m6.pt',
}


def init_folders():
    # Create directory
    os.makedirs(Path(DATASET_PATH) / DETECTION4CIL_PATH, exist_ok=True)

    # Copy images
    if not (Path(DATASET_PATH) / DETECTION4CIL_PATH / 'images').exists():
        print(f'Copying images. . .')
        shutil.copytree(
            src=Path(DATASET_PATH) / LOGODET_3K_NORMAL_PATH / "images",
            dst=Path(DATASET_PATH) / DETECTION4CIL_PATH / "images"
        )

    # Delete labels directory
    labels_path = Path(DATASET_PATH) / DETECTION4CIL_PATH / 'labels'
    shutil.rmtree(labels_path, ignore_errors=True)
    os.makedirs(labels_path, exist_ok=False)

    # Delete split files
    for file in ['train.txt', 'validation.txt', 'test.txt']:
        file_path = Path(DATASET_PATH) / DETECTION4CIL_PATH / file
        file_path.unlink(missing_ok=True)


def read_metadata():
    df_full = pd.read_pickle(Path(DATASET_PATH) / LOGODET_3K_NORMAL_PATH / 'metadata_full_images.pickle')
    df_cropped = pd.read_pickle(Path(DATASET_PATH) / LOGODET_3K_NORMAL_PATH / 'metadata_cropped_images.pickle')
    return df_full, df_cropped


def init_dataset():
    # Create dataset manager
    data_manager = DataManager(
        args['dataset'], args['shuffle'], args['seed'], args['init_cls'],
        args['increment'], data_augmentation=args['data_augmentation']
    )

    train = data_manager.get_dataset(indices=np.arange(0, args['init_cls']), source='train', mode='train')
    validation = data_manager.get_dataset(indices=np.arange(0, args['init_cls']), source='val', mode='test')
    test = data_manager.get_dataset(indices=np.arange(0, args['init_cls']), source='test', mode='test')

    return train, validation, test


def generate_split_file(filename, data, metadata):
    file_path = Path(DATASET_PATH) / DETECTION4CIL_PATH / filename

    metadata['cropped_image_path'] = metadata['cropped_image_path'].map(lambda x: str(x))
    full_images_list = []
    for image in tqdm(data.images, total=len(data.images)):
        # Get the name of the full image which contains the cropped image
        query = f'cropped_image_path == "{Path(image).name}"'
        full_image_row = metadata.query(query)
        full_image_filename = full_image_row['new_path'].iloc[0]
        full_image_path = Path(DATASET_PATH) / DETECTION4CIL_PATH / 'images' / full_image_filename
        full_images_list.append(str(full_image_path))

        # Append the cropped image bbox to the labels
        label_file_path = file_path.parents[0] / 'labels' / full_image_filename.with_suffix('.txt')
        with open(label_file_path, 'a') as f:
            image_label = full_image_row['yolo_label'].iloc[0]
            f.write(image_label + '\n')

    # Write path on file
    with open(file_path, 'w') as f:
        uniques_full_images = list(set(full_images_list))
        f.writelines('\n'.join(uniques_full_images))


def generate_labels(train, validation, test, df_cropped):
    # Generate train
    print('Generate training labels . . .')
    generate_split_file('train.txt', train, df_cropped)
    # Generate validation
    print('Generate validation labels . . .')
    generate_split_file('validation.txt', validation, df_cropped)
    # Generate test
    print('Generate test labels . . .')
    generate_split_file('test.txt', test, df_cropped)


def main():
    # Generate labels
    if not cmd_args.only_det:
        # Reset folders and files split
        init_folders()
        # Get dataset
        train, validation, test = init_dataset()
        # Get dataframe
        df_full, df_cropped = read_metadata()
        # Generate labels
        generate_labels(train, validation, test, df_cropped)

    # Generate yaml file
    yaml_filename = 'LogoDet-3K_CIL.yaml'
    with open(ROOT / yaml_filename, 'w') as file:
        yaml.dump(
            {
                'path': f'../{str(Path(DATASET_PATH) / DETECTION4CIL_PATH)}',
                'train': 'train.txt',
                'val': 'validation.txt',
                'test': 'test.txt',
                'nc': 1,
                'names': ['logo']
            },
            file
        )

    # Start training
    project = f'yolo-cil'
    run_name = f"yolo-CIL-{args['init_cls']}cls"

    command = f'python yolov5/train.py ' \
              f'--data {yaml_filename} ' \
              f'--img {args["img_size"]} ' \
              f'--batch {args["batch_size"]} ' \
              f'--epochs {args["epochs"]} ' \
              f'--weights yolov5/{args["weights"]} ' \
              f'--project {project} ' \
              f'--name {run_name}'

    print(command)
    
    if cmd_args.start_training:
        subprocess.run(command, shell=True)


if __name__ == '__main__':
    main()
