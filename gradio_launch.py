import PIL
import gradio
import argparse
import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from werkzeug.utils import secure_filename
import cv2
import time
from datetime import timedelta
import preprocess
from utils.data_loading import BasicDataset
from unet import UNet
from from2Dto3D import savenii
from unet.vit_seg_modeling import VisionTransformer as ViT_seg
from unet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from unet.att_unet import  AttUNet
from unet.att_cbam_unet import AttCBAMUNet
from zipfile import ZipFile
gradio.close_all()
os.makedirs('input/liver',exist_ok=True)
os.makedirs('output/liver',exist_ok=True)
os.makedirs('output/liver_nii',exist_ok=True)
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

# 设置允许的文件格式
ALLOWED_EXTENSIONS_LIVER = set(['nii', 'NII'])
ALLOWED_EXTENSIONS_LUNG = set(['png','PNG'])


def allowed_file(filename, type):
    if type == 'liver':
        return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS_LIVER
    else:
        return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS_LUNG

def predict_img(net,
                full_img,
                device,
                model,
                scale_factor,
                out_threshold=0.5,
                sz = 512):
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
    output = transforms.Resize([sz,sz])(output)
    return output

def output(file):
    return file

def predict2d(file, whichUNet, size):
    args = get_args()
    args.model = 'lung'
    if whichUNet == 'OriginUNet':
        args.UNet = 'AttCBAMUNet' 
    elif whichUNet == 'AttCBAMUNet':
        args.UNet = 'UNet' 
    else:
        args.UNet = whichUNet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.UNet == 'UNet':
        net = UNet(n_channels=1, n_classes=2, bilinear=True)
    elif args.UNet == "AttUNet":
        net = AttUNet(n_channels=1, n_classes=2)
    elif args.UNet == "AttCBAMUNet":
        net = AttCBAMUNet(n_channels=1, n_classes=2)
    elif args.UNet == 'TransUNet':
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = 2
        config_vit.n_skip = 3
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(device)
    args.load = 'saved_checkpoints/'+args.model+"_"+args.UNet+"_checkpoint_epoch50.pth"
    net.load_state_dict(torch.load(args.load, map_location=device))
    net.to(device=device)

    img = file.convert('L')
    mask = predict_img(net=net,
                        full_img=img,
                        scale_factor=args.scale,
                        out_threshold=args.mask_threshold,
                        device=device,
                        model = args.model, sz = size)
    out_filename = 'output.png'
    mask.save(out_filename)
    return out_filename, out_filename, str(size)+'x'+str(size)+' Predicted image is here for downloading!'

    
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def predict3d(nii, whichUNet):
    args = get_args()
    args.model = 'liver'
    args.UNet = 'UNet' if whichUNet == 'OriginUNet' else whichUNet

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model == 'liver':
        preprocess.process(nii, 'input/liver')

    if args.UNet == 'UNet':
        net = UNet(n_channels=3, n_classes=2, bilinear=True)
    elif args.UNet == "AttUNet":
        net = AttUNet(n_channels=3, n_classes=2)
    elif args.UNet == "AttCBAMUNet":
        net = AttCBAMUNet(n_channels=3, n_classes=2)
    elif args.UNet == 'TransUNet':
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = 2
        config_vit.n_skip = 3
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)
    args.load = 'saved_checkpoints/'+args.model+"_"+args.UNet+"_checkpoint_epoch50.pth"
    net.load_state_dict(torch.load(args.load, map_location=device))
    net.to(device=device)
   
    
    
    in_files = os.listdir('input/liver')
    samplemask = []
    totalsample = 0
    for in_file in in_files:
        data = np.load(os.path.join('input/liver',in_file))
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
            out_filename = in_file + '.' + str(id) + '.jpg'
            id = id + 1

            mask.save(os.path.join('output/liver', out_filename))
            if totalsample<64:
                samplemask.append(mask)
            totalsample += 1

    gridmask = 'gridmask.jpg'
    
    image_grid(samplemask, rows=8, cols=8).save(gridmask)
    file = savenii('output/liver','output/liver_nii')
    return gridmask, file, 'Nii files are provided for downloading, showing slices.'


inputs2d = [
gradio.Image(type='pil'),
gradio.Dropdown(['OriginUNet','AttUNet','AttCBAMUNet','TransUNet'], type="value"),
gradio.Slider(224, 512, value=224, label="Output_Size", info="选择输出图像大小"),
]

inputs3d = [
'text',
gradio.Dropdown(['OriginUNet','AttUNet','AttCBAMUNet','TransUNet'], type="value")
]

inputssyn = [
'text',
gradio.Dropdown(['OriginUNet','AttUNet','AttCBAMUNet','TransUNet'], type="value")
]

outputs2d = [
gradio.outputs.Image(label='分割结果',type='pil'),
gradio.outputs.File(label='可下载文件',),
gradio.outputs.Textbox(type = 'text', label='')
]
outputs3d = [
gradio.outputs.Image(label='分割结果',type='pil'),
'file',
'text'
]

outputssyn = [
gradio.outputs.File(label='原始数据'),
gradio.outputs.File(label='分割文件'),
'text'
]

def syna(input1,input2):
    return "case0008_img.nii.gz","case0008_pred.nii.gz",'Nii files are provided for downloading!'


demo = gradio.TabbedInterface(
    [
    gradio.Interface(
    fn=predict2d,
    inputs= inputs2d,
    outputs = outputs2d,
    examples= 
    [[os.path.join(os.path.dirname(__file__),"origin1.png"),
      'OriginUNet'],
     [os.path.join(os.path.dirname(__file__),"origin2.png"),
      'AttCBAMUNet'],
      
     [os.path.join(os.path.dirname(__file__),"origin3.png"),
      'TransUNet']]),
    
    

    gradio.Interface(
    fn=predict3d,
    inputs= inputs3d,
    outputs = outputs3d,
    examples=
    [
    [os.path.join(os.path.dirname(__file__),'input/liver_nii'),
      'AttCBAMUNet']
    ],
    cache_examples=False
    
    ),
    
    gradio.Interface(
    fn= syna,
    inputs= inputssyn,
    outputs = outputssyn,
    examples=
    [
    ['case0008.npy.h5',
      'TransUNet']
    ],
    cache_examples=False
    
    )
    ],
    
    ['肺部2D图像','肝脏3D图像','多器官3D图像'],title='Fly-Attention Segmentation')
demo.launch(debug = True,server_port=6006)
