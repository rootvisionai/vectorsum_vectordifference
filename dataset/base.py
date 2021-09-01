from __future__ import print_function
from __future__ import division
import torch
import PIL.Image
from skimage import io
import os
import numpy as np
from utils import random_crop

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, transform = None, crop_size=128, num_of_crops=7):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.crop_size = crop_size
        self.num_of_crops = num_of_crops

    def nb_classes(self):
        return len(self.classes)

    def __len__(self):
        return len(self.class_ids)

    def __getitem__(self, index):
        def img_load(index):
            im = PIL.Image.fromarray(io.imread(self.image_paths[index])[:,:,0:3])
            if len(list(im.split())) == 1:
                im = im.convert('RGB')
            if self.transform is not None:
                im = self.transform(im)
            return im

        im = img_load(index)
        target_int = self.class_ids[index]
        target_str = self.class_names[index]
        if self.mode == "train":
            image_out, bbox_out = random_crop(im, self.crop_size, self.num_of_crops)
            return image_out, bbox_out, target_int, target_str
        else:
            return im, target_int, target_str

    def get_label(self, index):
        return self.class_ids[index],self.class_names[index]

    def set_subset(self, I):
        self.class_ids = [self.class_ids[i] for i in I]
        self.class_names = [self.class_names[i] for i in I]
        self.image_ids = [self.image_ids[i] for i in I]
        self.image_paths = [self.image_paths[i] for i in I]
        
class Set(BaseDataset):
    def __init__(self, root, dpath, mode, transform = None):
        self.root = root.replace('\\','/') + dpath
        self.mode = mode if mode in ['train','generate'] else 'test'
        self.transform = transform
        
        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        metalist = []
        with open(os.path.join(self.root, '{}.txt'.format(mode)),'r') as f:
            lines = f.readlines()
            for line in lines:
                line_splitted = line.split(' ')
                if not len(line_splitted) == 0:
                    metalist.append(line_splitted)

        self.class_ids = []
        self.image_ids = []
        self.class_names = []
        self.image_paths = []
        self.angles = []
        self.label_map = {}

        for i, (image_id, class_id, class_name, path) in enumerate(metalist):
            if mode == 'generate':
                angle_range = range(0,1,1)
            elif mode == 'train':
                angle_range = range(0,1,1)
            elif mode == 'test':
                angle_range = range(0,1,1)
            for angle in angle_range:
                self.class_ids.append(int(class_id))
                self.image_ids.append(int(image_id))
                self.class_names.append(str(class_name))
                self.image_paths.append(self.root+path.replace('\n','').replace('./','/'))
                self.angles.append(angle)
        
        for pair in zip(self.class_names, self.class_ids):
            if pair[0] not in self.label_map:
                self.label_map[pair[0]] = int(pair[1])
        print("___ {} ___".format(len(np.unique(self.class_ids))))
        self.classes = range(len(np.unique(self.class_ids)))