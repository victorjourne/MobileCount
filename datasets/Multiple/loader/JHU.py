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
        self.gt_format = kwargs.get('JHU__gt_format', '.npz')
        self.transform = kwargs.get('JHU__transform', None)
        if self.gt_path is None:
            raise ValueError('Must specify `JHU__gt_path` parameter')
        self.folder = folder
        self.mode = mode
        self.dataset = self.read_index()

    def read_index(self):
        """
        Read all images position in WE Dataset
        """
        img_list = []
        if self.mode == 'test':
            for folder in self.val_folder:
                img_list += list(glob.glob(os.path.join(self.folder, f'{self.mode}', folder, 'images', '*')))
        else:
            img_list += list(glob.glob(os.path.join(self.folder, f'{self.mode}', 'images', '*')))

        json_data = {}
        for n, im in enumerate(img_list):
            print('im:',str(im))
            filename = pathlib.Path(im).stem
            gt_count = None
            json_data[n] = {
                "path_img": im,
                "path_gt": os.path.join(self.gt_path, filename) + self.gt_format,
                "gt_count": gt_count,
                "folder": self.folder
            }
        df = pd.DataFrame.from_dict(json_data, orient='index')
        print("df.shape:", df.shape)
        return df

    def load_gt(self, filename):

        density_map = pd.read_csv(filename, sep=',', header=None).values
        density_map = density_map.astype(np.float32, copy=False)

        self.check_density_map(density_map)
        return density_map
