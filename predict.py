import argparse
import logging
import os
import sys
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import preprocess
from utils.data_loading import BasicDataset
from unet import UNet
from from2Dto3D import savenii
from unet.vit_seg_modeling import VisionTransformer as ViT_seg
from unet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from unet.att_unet import  AttUNet
from unet.att_cbam_unet import AttCBAMUNet
def predict_img(net,
                full_img,
                device,
                model,
                scale_factor,
                out_threshold=0.5):
    net.eval()
    w = 0
    h = 0


    w, h = full_img.size
    img = full_img.resize([224,224], resample=Image.BICUBIC)
    img = np.asarray(img)
    if model =='lung':
        img = torch.from_numpy(img.copy()[np.newaxis,...])/255
    else:
        img = torch.from_numpy(img.copy())/255
        img = img.permute(2,0,1)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        output = torch.softmax(output, dim=1).argmax(dim=1)[0].float().cpu()
        


    output = transforms.ToPILImage()(output)
    output = transforms.Resize([512,512])(output)
    return output


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--load', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', default='input',metavar='INPUT', help='Filenames of input images')
    parser.add_argument('--output', '-o', default='output',metavar='INPUT', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--model', type=str, default=False, help='liver or lung')
    parser.add_argument('--UNet', type=str, default=False, help='which Unet model')
    parser.add_argument('--img_size', type=str, default=224, help='which Unet model')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

def main():
    args = get_args()
    in_files = os.path.join(args.input,args.model)
    out_files = args.output
    channels = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model == 'liver':
        preprocess.process(in_files, os.path.join(in_files, 'temp'))

    if args.UNet == 'UNet':
        net = UNet(n_channels=1 if args.model=='lung' else 3, n_classes=2, bilinear=True)
    elif args.UNet == "AttUNet":
        net = AttUNet(n_channels=1 if args.model == 'lung' else 3, n_classes=2)
    elif args.UNet == "AttCBAMUNet":
        net = AttCBAMUNet(n_channels=1 if args.model == 'lung' else 3, n_classes=2)
    elif args.UNet == 'TransUNet':
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = 2
        config_vit.n_skip = 3
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    else:
        print("Unknown UNet Model")
        sys.exit(-1)
    args.load = 'saved_checkpoints/'+args.model+"_"+args.UNet+"_checkpoint_epoch100.pth"
    net.load_state_dict(torch.load(args.load, map_location=device))
    logging.info(f'Model loaded from {args.load}')
   


    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')
    net.to(device=device)
    logging.info('Model loaded!')

    if args.model == 'liver':
        in_files = os.path.join(in_files, 'temp')
    files = os.listdir(in_files)

    outputpath = os.path.join(out_files, args.model)
    if os.path.exists(outputpath) == False:
        os.makedirs(outputpath)

    for file in files:
        path = os.path.join(in_files, file)

        if (args.model == 'liver'):
            if (file.endswith(".npz") == False):
                continue
            data = np.load(path)
            logging.info(f'\nPredicting image {file} ...')
            imgs = data["imgs"]
            id = 0
            for img in imgs:
                img = Image.fromarray(img.transpose(1,2,0),'RGB')
                
                mask = predict_img(net=net,
                                   full_img=img,
                                   model= args.model,
                                   scale_factor=args.scale,
                                   out_threshold=args.mask_threshold,
                                   device=device)
                out_filename = file + '.' + str(id) + '.jpg'
                id = id + 1
                mask.save(os.path.join(outputpath, out_filename))

                
                print(f'Mask saved to {os.path.join(out_files, out_filename)}')
            
        elif args.model == 'lung':
            img = Image.open(path).convert('L')

            mask = predict_img(net=net,
                               full_img=img,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device,
                               model = args.model)
            out_filename = file
            mask.save(os.path.join(outputpath, out_filename))
            print(f'Mask saved to {os.path.join(out_files, out_filename)}')
    savenii()
    
if __name__ == '__main__':
    main()