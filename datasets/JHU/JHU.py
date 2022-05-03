import numpy as np
import os
import random
from scipy import io as sio
import sys
import torch
from torch.utils import data
from PIL import Image, ImageOps
from scipy.sparse import load_npz
import pandas as pd

from config import cfg

class JHU(data.Dataset):
    def __init__(self, data_path, mode, main_transform=None, img_transform=None, gt_transform=None):
        self.img_path = data_path + '/images'
        #print('self.img_path:',self.img_path)
        gt_directory = '/workspace/home/jourdanfa/data/density_maps/jhu_crowd_v2.0/'+mode+'/'
        self.gt_path = gt_directory + '/dm'
        #print('self.gt_path:',self.gt_path)
        data_files = [filename for filename in os.listdir(self.img_path) \
                           if os.path.isfile(os.path.join(self.img_path,filename))]
        print('nb1:',len(data_files))
        self.data_files = []
        for filepath in data_files:
            #print('filepath:',filepath)
            filename = os.path.basename(filepath)
            gt_filepath = os.path.join(self.gt_path, filename + '.npz')
            #print('gt_filepath:',gt_filepath)
            if os.path.isfile(gt_filepath):
                self.data_files.append(filepath)
            else:
                print("KO")
                pass
        print('nb2:',len(self.data_files))
        self.num_samples = len(self.data_files) 
        self.main_transform=main_transform  
        self.img_transform = img_transform
        self.gt_transform = gt_transform   
    
    def __getitem__(self, index):
        fname = self.data_files[index]
        img, den = self.read_image_and_gt(fname)   
        #print('__getitem__ step1 img.size:', img.size, 'den.size:', den.size)
        if self.main_transform is not None:
            #print('__getitem__ main_transform')
            img, den = self.main_transform(img,den)
        #print('__getitem__ step2 img.size:', img.size, 'den.size:', den.size)
        if self.img_transform is not None:
            img = self.img_transform(img)
        #print('__getitem__ step3 img.size:', img.size, 'den.size:', den.size)
        if self.gt_transform is not None:
            den = self.gt_transform(den)
        #print('__getitem__ step4 img.size:', img.size, 'den.size:', den.size)
        return img, den

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self,fname):
        #print('img_path:',os.path.join(self.img_path,fname))
        img = Image.open(os.path.join(self.img_path,fname))
        if img.mode == 'L':
            img = img.convert('RGB')
        #print('read_image_and_gt img.size:', img.size)
        den_map_path = os.path.join(self.gt_path, fname + '.npz')
        #print('den_map_path:',den_map_path)
        density_map = load_npz(den_map_path).toarray()     
        den = density_map.astype(np.float32, copy=False)
        den = Image.fromarray(den)
        #print('read_image_and_gt den.size:', den.size)
        return img, den    

    def get_num_samples(self):
        return self.num_samples       
            
    def load_sparse(self, filename):
        return load_npz(filename).toarray()
