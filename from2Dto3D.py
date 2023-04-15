import os
from torchvision.utils import make_grid
import SimpleITK as sitk
import glob
import numpy as np
from PIL import Image
import cv2
import torch
import matplotlib.pyplot as plt  # plt 用于显示图片
from einops import rearrange


def save_array_as_nii_volume(data, filename, reference_name=None):
    """
    save a numpy array as nifty image
    inputs:
        data_lung: a numpy array with shape [Depth, Height, Width]
        filename: the ouput file name
        reference_name: file name of the reference image of which affine and header are used
    outputs: None
    """
    img = sitk.GetImageFromArray(data)
    if (reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, filename)

def savenii(image_path = 'output/liver',output_path= 'output/liver_nii'):

    
    image_arr = os.listdir(image_path)

    allImg = np.zeros([123, 512, 512], dtype='uint8')
    for i in range(len(image_arr)):
        single_image_name = os.path.join(image_path,image_arr[i])
        img_as_img = Image.open(single_image_name)
        # img_as_img.show()
        img_as_np = np.asarray(img_as_img)
        allImg[i, :, :] = img_as_np
    
    # np.transpose(allImg,[2,0,1])
    save_array_as_nii_volume(allImg, output_path+'/output.nii')
    return output_path+'/output.nii'
