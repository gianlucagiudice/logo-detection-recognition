import os
import shutil
import xml.etree.ElementTree as ET
from zipfile import ZipFile
import pandas as pd
import numpy as np
import argparse
import sys
import cv2
import tqdm

from config import *

sys.path.append('./yolov5')

from yolov5.utils.general import Path

SEED = 830694

DATASET_ROOT = 'LogoDet-3K'


parser = argparse.ArgumentParser(description='Download LogoDet-3k.')


parser.add_argument('--train-split', type=float, default=DATASET_TRAIN_SPLIT,
                    help='Training split ratio.')

parser.add_argument('--validation-split', type=float, default=DATASET_VALIDATION_SPLIT,
                    help='Validation split ratio.')

parser.add_argument('--test-split', type=float, default=DATASET_TEST_SPLIT,
                    help='Test split ratio.')

parser.add_argument('--dataset-type', type=str, required=True, default='small',
                    choices=['small', 'sample', 'normal'],
                    help='Type of LogoDet-3K dataset {small/sample/normal}.')

parser.add_argument('--sampling-fraction', type=float, required=True, default=1,
                    help='Number of categories to sample.')

parser.add_argument('--only-sample', type=bool, required=False, default=False, action=argparse.BooleanOptionalAction,
                    help='Number of categories to sample.')

parser.add_argument('--only-top', type=bool, required=False, default=False, action=argparse.BooleanOptionalAction,
                    help='Number of categories to sample.')

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
assert sum([train_split, validation_split, test_split]) == 1

url_dataset = type2url[args.dataset_type]
dir_path = Path(f'{DATASET_PATH}/{type2path[args.dataset_type]}')

original_path = dir_path.joinpath(DATASET_ROOT)
images_path = dir_path.joinpath(DATASET_IMAGES_PATH)
labels_path = dir_path.joinpath(DATASET_LABELS_PATH)
cropped_path = dir_path.joinpath(DATASET_CROPPED_PATH)

sampling_fraction = float(args.sampling_fraction)


def generate_yolo_labels(xml_file_path: Path, obj_class=0):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    image_width, image_hight = int(root.find('size/width').text), int(root.find('size/height').text)

    objs = [
        dict(xmin=int(obj.find('bndbox/xmin').text),
             ymin=int(obj.find('bndbox/ymin').text),
             xmax=int(obj.find('bndbox/xmax').text),
             ymax=int(obj.find('bndbox/ymax').text),
             brand=obj.find('name').text)
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
    # Extract brands
    brands = [obj['brand'] for obj in objs]
    return '\n'.join(lines), objs, brands, tree.find('filename').text


def create_dataset_split(split_filename: Path, images) -> None:
    with open(split_filename, 'w') as f:
        np.random.shuffle(images)
        f.writelines('\n'.join([str(x) for x in images]))


# Check if all the files already exists
if all([
    dir_path.joinpath('train.txt').exists(),
    dir_path.joinpath('test.txt').exists(),
    dir_path.joinpath('validation.txt').exists(),
    images_path.exists(), labels_path.exists(), cropped_path.exists(),
    dir_path.joinpath(METADATA_FULL_IMAGE_PATH).exists(),
    dir_path.joinpath(METADATA_CROPPED_IMAGE_PATH).exists(),
    not args.only_sample
]):
    quit(0)


if args.only_sample:
    # Read dataframes from file
    df_metadata_full = pd.read_pickle(str(dir_path.joinpath(METADATA_FULL_IMAGE_PATH)))
    df_metadata_cropped = pd.read_pickle(str(dir_path.joinpath(METADATA_CROPPED_IMAGE_PATH)))
else:
    # Reset Directories
    shutil.rmtree(images_path, ignore_errors=True)
    shutil.rmtree(labels_path, ignore_errors=True)
    shutil.rmtree(cropped_path, ignore_errors=True)
    shutil.rmtree(dir_path.joinpath('train.txt'), ignore_errors=True)
    shutil.rmtree(dir_path.joinpath('test.txt'), ignore_errors=True)
    shutil.rmtree(dir_path.joinpath('validation.txt'), ignore_errors=True)
    shutil.rmtree(original_path, ignore_errors=True)
    dir_path.joinpath(METADATA_FULL_IMAGE_PATH).unlink(missing_ok=True)
    dir_path.joinpath(METADATA_CROPPED_IMAGE_PATH).unlink(missing_ok=True)

    Path.mkdir(Path(DATASET_PATH), exist_ok=True)
    Path.mkdir(dir_path, exist_ok=True)
    Path.mkdir(images_path, exist_ok=False)
    Path.mkdir(labels_path, exist_ok=False)
    Path.mkdir(cropped_path, exist_ok=False)

    # Download
    os.system(f"wget -nc '{url_dataset}' -O {dir_path / 'out.zip'}")
    ZipFile(dir_path / 'out.zip').extractall(path=dir_path)

    # Get all files in a given directory
    get_all_files = lambda folder_path: [Path(Path(currentpath).joinpath(file))
                                         for currentpath, folders, files in os.walk(folder_path)
                                         for file in files]
    # Read all images
    all_images = [x for x in get_all_files(original_path) if x.suffix == '.jpg']

    # Metadata
    df_metadata_full = pd.DataFrame(columns=['original_path', 'new_path', 'filename', 'category'])
    df_metadata_cropped = pd.DataFrame(columns=['cropped_image_path', 'original_path', 'new_path', 'category', 'brand'])


    for i, im_path in tqdm.tqdm(enumerate(all_images), total=len(all_images)):
        # Generate new name
        filename = Path(f"{'0' * (len(str(len(all_images))) - len(str(i)))}{str(i)}.jpg")
        # Generate label file
        yolo_label_content, objects, brands, original_filename = generate_yolo_labels(im_path.with_suffix('.xml'))
        # Add metadata full image
        *_, category, brand, _ = os.path.normpath(im_path).split(os.path.sep)
        new_row_full_image = dict(
            original_path=str(im_path),
            new_path=str(images_path / filename),
            filename=original_filename,
            category=category,
            brand=brand
        )
        df_metadata_full = pd.concat([df_metadata_full, pd.DataFrame.from_records([new_row_full_image])])
        # Crop all objects
        img = cv2.imread(str(im_path))

        for brand, obj in zip(brands, objects):
            # Crop the logo
            cropped_image = img[obj['ymin']:obj['ymax'], obj['xmin']:obj['xmax'], :]
            # Generate filename
            padding = '0' * (6 - len(str(len(df_metadata_cropped))))
            cropped_filename = Path(f"{padding + str(len(df_metadata_cropped))}.jpg")
            # Save image
            try:
                cv2.imwrite(str(cropped_path / cropped_filename), cropped_image)
            except Exception as e:
                print(e)
                print(f'Error: {im_path} - {obj}')
                continue
            # Add metadata
            new_row_cropped_image = dict(
                cropped_image_path=cropped_filename,
                original_path=im_path,
                new_path=filename,
                brand=brand,
                category=category
            )
            df_metadata_cropped = pd.concat([df_metadata_cropped, pd.DataFrame.from_records([new_row_cropped_image])])

        # Move image
        im_path.rename(images_path.joinpath(filename))
        # Create file
        with open(labels_path.joinpath(filename.with_suffix('.txt')), 'w') as f:
            f.writelines(yolo_label_content)

    # Export dataframe
    df_metadata_full.to_pickle(str(dir_path.joinpath(METADATA_FULL_IMAGE_PATH)))
    df_metadata_cropped.to_pickle(str(dir_path.joinpath(METADATA_CROPPED_IMAGE_PATH)))

# Create train/validation/test
np.random.seed(SEED)

# Sample classes
unique_brands = df_metadata_full['brand'].unique()

# Convert to sampling fraction
if sampling_fraction > 1:
    sampling_fraction = sampling_fraction / len(unique_brands)

if args.only_top:
    # Consider only the classes with the highest number of sample
    categories_frequency = df_metadata_full.groupby(by=['brand']).count()['original_path']
    categories_sorted_by_frequency = list(map(lambda x: x[0],
                                              sorted(zip(categories_frequency.index, categories_frequency),
                                                     key=lambda x: x[1], reverse=True)))
    unique_brands = categories_sorted_by_frequency
else:
    # Shuffle the classes randomly
    np.random.shuffle(unique_brands)

sampled_classes = unique_brands[:round(sampling_fraction * len(unique_brands))]
print(f'Number of sampled classes: {len(sampled_classes)} '
      f'({len(sampled_classes) / len(unique_brands) * 100:.4}%)')

# Sample instances
sampled_instances = df_metadata_full[df_metadata_full['brand'].isin(sampled_classes)]['new_path'].values
np.random.shuffle(sampled_instances)
split_ids = [round(len(sampled_instances)*train_split), round(len(sampled_instances)*(train_split + validation_split))]
training_data, validation_data, test_data = np.split(sampled_instances, split_ids)
print(f'Number of sampled instances: {len(sampled_instances)} '
      f'({len(sampled_instances) / len(df_metadata_full) * 100:.4}%)\n'
      f'Training split info: [\n'
      f'\ttraining = {len(training_data)} ({len(training_data) / len(sampled_instances)*100:.4}%);\n'
      f'\tvalidation = {len(validation_data)} ({len(validation_data) / len(sampled_instances)*100:.4}%);\n'
      f'\ttest = {len(test_data)} ({len(test_data) / len(sampled_instances) * 100:.4}%)\n]')

create_dataset_split(dir_path.joinpath('train.txt'), training_data)
create_dataset_split(dir_path.joinpath('validation.txt'), validation_data)
create_dataset_split(dir_path.joinpath('test.txt'), test_data)

# Remove old directory
shutil.rmtree(original_path, ignore_errors=True)
