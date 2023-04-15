
import os


import numpy as np
import nibabel as nib
from tqdm import tqdm
import scipy
import matplotlib.pyplot as plt

import utils.util as util
from utils.config import cfg
import utils.util as util

import argparse
import aug
def process(input=None,output=None):
    # 1. 创建输出路径，删除非空的summary表格

    if input is not None:
        cfg.DATA.INPUTS_PATH = input
    if output is not None:
        cfg.DATA.PREP_PATH = output
    if cfg.PREP.PLANE == "xy" and not os.path.exists(cfg.DATA.PREP_PATH):
        os.makedirs(cfg.DATA.PREP_PATH)
    volumes = util.listdir(cfg.DATA.INPUTS_PATH)

    vol_npz = []
    npz_count = 0
    thick = 1
    pbar = tqdm(range(len(volumes)-1), desc="数据处理中")
    for i in range(len(volumes)):
        if(volumes[i].endswith(".nii")==False):
            continue
        pbar.set_postfix(filename=volumes[i] + " " + volumes[i])
        pbar.update(1)
        volf = nib.load(os.path.join(cfg.DATA.INPUTS_PATH, volumes[i]))
        volume = volf.get_fdata()
        volume = volume.astype(np.float16)

        crop_size = list(cfg.PREP.SIZE)
        for ind in range(3):
            if crop_size[ind] == -1:
                crop_size[ind] = volume.shape[ind]
        volume = aug.crop(volume, None, crop_size)

        # 开始切片
        for frame in range(1, volume.shape[2] - 1):
            vol = volume[:, :, frame - thick : frame + thick + 1]
            vol = np.swapaxes(vol, 0, 2)
            vol_npz.append(vol.copy())
            print("{} 片满足，当前共 {}".format(frame, len(vol_npz)))

            if len(vol_npz) == cfg.PREP.BATCH_SIZE or (
                i == (len(volumes) - 1) and frame == volume.shape[2] - 1
            ):
                imgs = np.array(vol_npz)
                print(imgs.shape)
                file_name = "{}_{}_f{}-{}".format(
                    cfg.DATA.NAME, cfg.PREP.PLANE, cfg.PREP.FRONT, npz_count
                )
                file_path = os.path.join(cfg.DATA.PREP_PATH, file_name)
                np.savez(file_path, imgs=imgs)
                vol_npz = []
                npz_count += 1
        
    pbar.close()


if __name__ == "__main__":
    process()