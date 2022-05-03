import glob
import os
import pathlib

import numpy as np
import pandas as pd

from .dynamics import CustomDataset


class CustomJHU(CustomDataset):
    def __init__(self, folder, mode, **kwargs):
        super().__init__()
        self.gt_path = kwargs.get('JHU__gt_path')
        self.gt_format = kwargs.get('JHU__gt_format', '.csv')
        self.transform = kwargs.get('JHU__transform', None)
        if self.gt_path is None:
            raise ValueError('Must specify `JHU__gt_path` parameter')
        self.folder = folder
        self.mode = mode
        # Ajout des images 'val' dans 'train' pour augmenter la proportion train/test 65%/35%
        # self.add_val_in_train = False
        self.dataset = self.read_index()

    def read_index(self):
        """
        Read all images position in JHU Dataset
        """
        img_list = list(glob.glob(
            os.path.join(self.folder, f'{self.mode}', 'images', '*.jpg')))  # train 2272 images / test 1600 images
        # if self.add_val_in_train and self.mode == 'train':
        #     img_list += list(glob.glob(os.path.join(self.folder, 'val', 'images', '*.jpg')))  # 501 images

        json_data = {}
        for n, im in enumerate(img_list):
            filename = pathlib.Path(im).stem
            gt_count = None
            sub_directory = self.mode
            # if self.add_val_in_train and '/val/images/' in im:
            #     sub_directory = 'val'
            path_gt = os.path.join(self.gt_path, sub_directory, 'den', filename + '.csv')
            json_data[n] = {
                "path_img": im,
                "path_gt": path_gt,
                "gt_count": gt_count,
                "folder": self.folder
            }
        df = pd.DataFrame.from_dict(json_data, orient='index')
        print(f'CustomJHU - mode:{self.mode} - df.shape:{df.shape}')
        return df

    def load_gt(self, filename):

        density_map = pd.read_csv(filename, sep=',', header=None).values
        density_map = density_map.astype(np.float32, copy=False)
        # print('density_map:', density_map.sum())

        self.check_density_map(density_map)
        return density_map
