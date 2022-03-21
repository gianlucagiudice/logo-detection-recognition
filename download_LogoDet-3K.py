import os
import shutil
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
import argparse
import sys
from config import *

sys.path.append('./yolov5')

from yolov5.utils.general import Path

SEED = 830694

DATASET_ROOT = 'LogoDet-3K'

dir_path = Path(f'{DATASET_PATH}/LogoDet-3K_small')
url_dataset = 'https://drive.google.com/uc?export=download&id=1D7BlreEbpyaKh7-O1GJz96EYFSyiYWa8'


parser = argparse.ArgumentParser(description='Download LogoDet-3k.')


parser.add_argument('--train_split', type=float, default=DATASET_TRAIN_SPLIT,
                    help='Training split ratio.')

parser.add_argument('--validation_split', type=float, default=DATASET_VALIDATION_SPLIT,
                    help='Validation split ratio.')

parser.add_argument('--test_split', type=float, default=DATASET_TEST_SPLIT,
                    help='Test split ratio.')

parser.add_argument('--dataset_type', type=str, required=True, default='small',
                    choices=['small', 'sample', 'normal'],
                    help='Type of LogoDet-3K dataset {small/sample/normal}.')

args = parser.parse_args()

type2url = dict(
    small=LOGODET_3K_SMALL_URL,
    sample=LOGODET_3K_SAMPLE_URL,
    normal=LOGODET_3K_NORMAL_URL
)

type2path = dict(
    small=LOGODET_3K_SMALL_PATH,
    sample=LOGODET_3K_SAMPLE_PATH,
    normal=LOGODET_3K_NORMAL_PATH
)

train_split, validation_split, test_split = args.train_split, args.validation_split, args.test_split
url_dataset = type2url[args.dataset_type]
dir_path = Path(f'{DATASET_PATH}/{type2path[args.dataset_type]}')

original_path = dir_path.joinpath(DATASET_ROOT)
images_path = dir_path.joinpath(DATASET_IMAGES_PATH)
labels_path = dir_path.joinpath(DATASET_LABELS_PATH)


def generate_yolo_labels(xml_file_path: Path, obj_class=0) -> str:
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    image_width, image_hight = int(root.find('size/width').text), int(root.find('size/height').text)

    objs = [
        dict(xmin=int(obj.find('bndbox/xmin').text),
             ymin=int(obj.find('bndbox/ymin').text),
             xmax=int(obj.find('bndbox/xmax').text),
             ymax=int(obj.find('bndbox/ymax').text)
             )
        for obj in root.findall('object')
    ]

    lines = []
    for obj in objs:
        w_middle = (obj['xmin'] / image_width) + ((obj['xmax'] - obj['xmin']) / image_width / 2)
        h_middle = (obj['ymin'] / image_hight) + ((obj['ymax'] - obj['ymin']) / image_hight / 2)
        obj_width = (obj['xmax'] - obj['xmin']) / image_width
        obj_hight = (obj['ymax'] - obj['ymin']) / image_hight
        lines.append(f'{obj_class} '
                     f'{round(w_middle, 6)} '
                     f'{round(h_middle, 6)} '
                     f'{round(obj_width, 6)} '
                     f'{round(obj_hight, 6)}')

    return '\n'.join(lines)


def create_dataset_split(split_filename: Path, images) -> None:
    with open(split_filename, 'w') as f:
        np.random.shuffle(images)
        f.writelines('\n'.join([str(dir_path / DATASET_IMAGES_PATH / x) for x in images]))


# Check if all the files already exists
if all([
    dir_path.joinpath('train.txt').exists(),
    dir_path.joinpath('test.txt').exists(),
    dir_path.joinpath('validation.txt').exists(),
    images_path.exists(), labels_path.exists()
]):
    quit(0)

# Reset Directories
shutil.rmtree(images_path, ignore_errors=True)
shutil.rmtree(labels_path, ignore_errors=True)
shutil.rmtree(dir_path.joinpath('train.txt'), ignore_errors=True)
shutil.rmtree(dir_path.joinpath('test.txt'), ignore_errors=True)
shutil.rmtree(dir_path.joinpath('validation.txt'), ignore_errors=True)
shutil.rmtree(original_path, ignore_errors=True)


Path.mkdir(Path(DATASET_PATH), exist_ok=True)
Path.mkdir(dir_path, exist_ok=True)
Path.mkdir(images_path, exist_ok=False)
Path.mkdir(labels_path, exist_ok=False)

# Download
os.system(f"wget '{url_dataset}' -O {dir_path / 'out.zip'}")
ZipFile(dir_path / 'out.zip').extractall(path=dir_path)

# Get all files in a given directory
get_all_files = lambda folder_path: [Path(Path(currentpath).joinpath(file))
                                     for currentpath, folders, files in os.walk(folder_path)
                                     for file in files]
# Read all images
all_images = [x for x in get_all_files(original_path) if x.suffix == '.jpg']

# Create new names for images and convert labels
tot_images = len(all_images)
padding_digits = len(str(tot_images))
new_filenames = []

for i, im_path in enumerate(all_images):
    # Generate new name
    filename = Path(f"{'0' * (padding_digits - len(str(i))) + str(i)}.jpg")
    new_filenames.append(filename)
    # Move image
    im_path.rename(images_path.joinpath(filename))
    # Generate label file
    yolo_label_content = generate_yolo_labels(im_path.with_suffix('.xml'))
    # Create file
    with open(labels_path.joinpath(filename.with_suffix('.txt')), 'w') as f:
        f.writelines(yolo_label_content)

new_filenames = np.array(new_filenames)

# Create train/validation/test

np.random.seed(SEED)

split = np.random.choice(
    [0, 1, 2], p=[train_split, validation_split, test_split], size=len(new_filenames)
)

create_dataset_split(dir_path.joinpath('train.txt'), new_filenames[split == 0])
create_dataset_split(dir_path.joinpath('validation.txt'), new_filenames[split == 1])
create_dataset_split(dir_path.joinpath('test.txt'), new_filenames[split == 2])

# Remove old directory
shutil.rmtree(original_path)
