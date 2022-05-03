import os

import numpy as np
from PIL import Image
from scipy.sparse import load_npz
from torch.utils import data


class JHU(data.Dataset):
    def __init__(self, data_path, mode, main_transform=None, img_transform=None, gt_transform=None):
        self.img_path = data_path + '/images'
        gt_directory = '/workspace/home/jourdanfa/data/density_maps/jhu_crowd_v2.0/' + mode + '/'
        self.gt_path = gt_directory + '/dm'
        data_files = [filename for filename in os.listdir(self.img_path) \
                      if os.path.isfile(os.path.join(self.img_path, filename))]
        print('nb1:', len(data_files))
        self.data_files = []
        for filepath in data_files:
            filename = os.path.basename(filepath)
            gt_filepath = os.path.join(self.gt_path, filename + '.npz')
            if os.path.isfile(gt_filepath):
                self.data_files.append(filepath)
            else:
                print("KO")
                pass
        print('nb2:', len(self.data_files))
        self.num_samples = len(self.data_files)
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

    def __getitem__(self, index):
        fname = self.data_files[index]
        img, den = self.read_image_and_gt(fname)
        if self.main_transform is not None:
            img, den = self.main_transform(img, den)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            den = self.gt_transform(den)
        return img, den

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self, fname):
        img = Image.open(os.path.join(self.img_path, fname))
        if img.mode == 'L':
            img = img.convert('RGB')
        den_map_path = os.path.join(self.gt_path, fname + '.npz')
        density_map = load_npz(den_map_path).toarray()
        den = density_map.astype(np.float32, copy=False)
        den = Image.fromarray(den)
        return img, den

    def get_num_samples(self):
        return self.num_samples

    def load_sparse(self, filename):
        return load_npz(filename).toarray()
