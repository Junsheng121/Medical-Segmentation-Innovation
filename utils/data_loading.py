import logging
import os
from os import listdir
from os.path import splitext
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

def CustomDataset(model):
    if model == 'liver':
        return LiverDataset
    elif model == 'lung':
        return BasicDataset
class BasicDataset(Dataset):
    def __init__(self, dirs, size: int = 224, mask_suffix: str = '_mask'):

        self.images_dir = Path(dirs+'/imgs/')
        self.masks_dir = Path(dirs+'/masks/')
        self.size = size
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(self.images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {self.images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, mask, size):
        w, h = pil_img.size
        if(mask is not None):
            assert (pil_img.size == mask.size)
        pil_img = pil_img.resize((size, size), resample= Image.BICUBIC)
        if(mask is not None):
            mask =  mask.resize((size, size), resample=Image.NEAREST)
        img_ndarray = np.asarray(pil_img)
        if (mask is not None):
            mask_ndarray = np.asarray(mask)

        if img_ndarray.ndim == 2:
            img_ndarray = img_ndarray[np.newaxis, ...]
        img_ndarray = img_ndarray / 255
        if (mask is not None):
            return img_ndarray, mask_ndarray/255
        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0]).convert('L')
        img = self.load(img_file[0]).convert('L')

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img, mask = self.preprocess(img, mask, self.size)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }




class LiverDataset(Dataset):
    def __init__(self, dir, size=224):
        super().__init__()
        files = os.listdir(dir)
        imglist = []
        masklist = []
##        
        files=files[0:10]
##
        for file in files:
            data = np.load(os.path.join(dir,file))
            imglist.append(data["imgs"])
            masklist.append(data["labs"])

        imglist = np.array(imglist)
        _,_, _, sizeh, sizew = imglist.shape
        self.size = size
        self.imglist = np.reshape(imglist,[-1,3,sizeh,sizew])
        masklist = np.array(masklist)
        self.masklist = np.reshape(masklist,[-1,1,sizeh,sizew])
        self.totalindex=imglist.shape[0]
    def preprocess(cls, pil_img, mask, size):
        pil_img = Image.fromarray(pil_img.transpose(1,2,0),'RGB')
        mask = np.uint8(mask[0])
        mask = Image.fromarray(mask*255)
        w, h = pil_img.size
        if(mask is not None):
            assert (pil_img.size == mask.size)
        pil_img = pil_img.resize((size, size), resample= Image.BICUBIC)
        if(mask is not None):
            mask =  mask.resize((size, size), resample=Image.NEAREST)
        img_ndarray = np.asarray(pil_img)
        if (mask is not None):
            mask_ndarray = np.asarray(mask)


        img_ndarray = img_ndarray.transpose(2,0,1)
        img_ndarray = img_ndarray / 255
        if (mask is not None):
            return img_ndarray, mask_ndarray/255
        return img_ndarray
    def __len__(self):
        return self.totalindex


    def __getitem__(self, idx):
        img = self.imglist[idx]
        mask = self.masklist[idx]

        img, mask = self.preprocess(img, mask, self.size)
        return {
            'image': torch.as_tensor(img).float().contiguous(),
            'mask': torch.as_tensor(mask).long().contiguous()
        }


