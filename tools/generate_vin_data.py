"""
Generate Dataset for VIN(Vehicle Identification Number)
"""

import os
import argparse
import random

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(description='Generate dataset for VIN(vehicle identification number')
    parser.add_argument('--root_folder', type=str, required=True, help='the root folder of labeled VIN data')
    parser.add_argument('--save_folder',
                        type=str,
                        default='./data',
                        help='the folder to save the train and val image list')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='the validation image ratio in all images')
    args = parser.parse_args()
    print(args)

    # get root and save folder
    root_folder = os.path.abspath(args.root_folder)
    save_folder = os.path.abspath(args.save_folder)
    print(f'root folder = {root_folder}, save folder = {save_folder}')

    # get the images and labels folder of VIN dataset
    images_folder = os.path.join(root_folder, 'images')
    labels_folder = os.path.join(root_folder, 'labels')

    # list all images
    all_images = []
    for f in os.listdir(images_folder):
        if f.endswith('jpg') or f.endswith('png') or f.endswith('bmp'):
            all_images.append(os.path.join(images_folder, f))

    # select images to train and val
    val_num = int(len(all_images) * args.val_ratio)
    val_images = random.sample(all_images, val_num)
    train_images = set(all_images) - set(val_images)
    print(f'all images number = {len(all_images)}'
          f', train images number = {len(train_images)}, val images number = {len(val_images)}')

    # write train and val images list to file
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    train_file = os.path.join(save_folder, 'train.txt')
    val_file = os.path.join(save_folder, 'val.txt')
    with open(train_file, 'w') as f:
        for v in train_images:
            f.write(f'{v}\n')
    with open(val_file, 'w') as f:
        for v in val_images:
            f.write(f'{v}\n')