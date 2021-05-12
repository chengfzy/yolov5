import xml.etree.ElementTree as ET
import os
import argparse
import random
from os.path import join


class VocYoloConverter:
    """Parse Pascal VOC dataset(labels) to Yolo format, and save image and labels path to file"""

    def __init__(self, voc_folder, sets, classes):
        self.voc_folder = voc_folder
        self.sets = sets
        self.classes = classes

    def convert(self, save_folder):
        """Convert dataset to YOLO format, then save image and label folders to file"""
        # make save folder if don't exist
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for year, image_set in sets:
            if not os.path.exists('{}/VOCdevkit/VOC{}/labels/'.format(self.voc_folder, year)):
                os.makedirs('{}/VOCdevkit/VOC{}/labels/'.format(self.voc_folder, year))

            # parse image ids(names) for each classes
            image_ids = []
            for c in self.classes:
                file = '{}/VOCdevkit/VOC{}/ImageSets/Main/{}_{}.txt'.format(self.voc_folder, year, c, image_set)
                with open(file) as f:
                    for line in f.readlines():
                        # -1: negative, 0: difficult, 1: positive
                        image_class_pair = line.strip().split()
                        if len(image_class_pair) != 2:
                            raise AttributeError('Error in this line: {}'.format(line))
                        if image_class_pair[1] == '1':
                            image_ids.append(image_class_pair[0])

            # remove duplicated image ids
            image_ids = list(set(image_ids))

            # parse annotations, save image and labels folder to file
            list_file = open('{}/{}_{}.txt'.format(save_folder, year, image_set), 'w')
            for image_id in image_ids:
                list_file.write('{}/VOCdevkit/VOC{}/JPEGImages/{}.jpg\n'.format(self.voc_folder, year, image_id))
                self.convert_annotation(year, image_id)
            list_file.close()

    def convert_annotation(self, year, image_id):
        """Parse annotation file to obtain labels and bounding box"""
        in_file = open('{}/VOCdevkit/VOC{}/Annotations/{}.xml'.format(self.voc_folder, year, image_id))
        out_file = open('{}/VOCdevkit/VOC{}/labels/{}.txt'.format(self.voc_folder, year, image_id), 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            box = obj.find('bndbox')
            b = (float(box.find('xmin').text), float(box.find('xmax').text), float(box.find('ymin').text),
                 float(box.find('ymax').text))
            bb = self.convert_box((w, h), b)
            out_file.write(str(cls_id) + ' ' + ' '.join([str(a) for a in bb]) + '\n')

    @staticmethod
    def convert_box(size, box):
        """Convert bounding box to [0, 1]"""
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--voc_folder',
                        action='store',
                        default='/home/jeffery/Documents/Code/datasets/VOC',
                        help='PASCAL VOC folder')
    parser.add_argument('-s',
                        '--save_folder',
                        action='store',
                        default='/home/jeffery/Documents/Code/datasets/VOC',
                        help='save folder')
    parser.add_argument('--show_result', action='store', type=bool, default=True, help='show result randomly')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
    sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
    classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    # classes = ['cat']

    converter = VocYoloConverter(args.voc_folder, sets, classes)
    converter.convert(save_folder=args.save_folder)

    if args.show_result:
        import cv2

        image_lists = open('{}/{}_{}.txt'.format(args.save_folder, sets[0][0], sets[0][1]), 'r').read().strip().split()
        image_file = random.choice(image_lists)
        # image_file = image_lists[0]
        image_label_file = image_file.replace('JPEGImages', 'labels').replace('.jpg', '.txt')

        # read image
        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        width = image.shape[1]
        height = image.shape[0]
        # parse labels
        labels = open(image_label_file).read().strip().split()
        class_name = classes[int(labels[0])]
        center_x = float(labels[1]) * width
        center_y = float(labels[2]) * height
        box_width = float(labels[3]) * width
        box_height = float(labels[4]) * height
        x0 = int(center_x - box_width / 2)
        y0 = int(center_y - box_height / 2)
        x1 = int(center_x + box_width / 2)
        y1 = int(center_y + box_height / 2)
        # draw box
        image = cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 1)
        # show
        cv2.imshow('Images', image)
        cv2.waitKey()
