import glob
import numpy as np
import os
import pandas as pd
import pathlib
from scipy.sparse import load_npz

from .dynamics import CustomDataset


class CustomJHU(CustomDataset):
    def __init__(self, folder, mode, **kwargs):
        super().__init__()
        self.gt_path = kwargs.get('JHU__gt_path')
        self.gt_format = kwargs.get('JHU__gt_format', '.npz')
        self.transform = kwargs.get('JHU__transform', None)
        if self.gt_path is None:
            raise ValueError('Must specify `JHU__gt_path` parameter')
        self.dataset_weight = kwargs.get('JHU__dataset_weight', 1)
        print('dataset_weight:',self.dataset_weight)
        self.folder = folder
        self.mode = mode
        self.dataset = self.read_index()

    def read_index(self):
        """
        Read all images position in JHU Dataset
        """
        img_list = list(glob.glob(
            os.path.join(self.folder, f'{self.mode}', 'images', '*.jpg')))  # train 2272 images / test 1600 images

        json_data = {}
        count = 0
        for n, im in enumerate(img_list):
            filename = os.path.basename(im)
            gt_count = None
            sub_directory = self.mode
            path_gt = os.path.join(self.gt_path, self.mode, 'dm', filename + '.npz')
            if os.path.isfile(path_gt):
                json_data[n] = {
                    "path_img": im,
                    "path_gt": path_gt,
                    "gt_count": gt_count,
                    "folder": self.folder,
                    "sample_weight": self.dataset_weight,
                }
                count += 1
            else:
                # no .npZ file means no correct points on image
                print("KO")
                pass
        print('nb1:', n + 1)
        print('nb2:', count)
        df = pd.DataFrame.from_dict(json_data, orient='index')
        print(f'CustomJHU - mode:{self.mode} - df.shape:{df.shape}')
        return df

    def load_gt(self, filename):

        density_map_path = os.path.join(self.gt_path, filename)
        density_map = load_npz(density_map_path).toarray()
        density_map = density_map.astype(np.float32, copy=False)

        self.check_density_map(density_map)
        return density_map
