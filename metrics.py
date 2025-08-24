#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os



from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim, msssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import sys
from arguments import ModelParams, PipelineParams, OptimizationParams
import imageio
import os
import numpy as np
import cv2
import csv
import torch 
import lpips as lp
from utils.image_utils import psnr
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'



tonemap = lambda x : (np.log(np.clip(x, 0, 1) * 5000 + 1 ) / np.log(5000 + 1)).astype(np.float32)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
# 展示LDR
def show_image(image,idx):
    show_image = Image.fromarray(to8b(image))
    show_image.save(f'rgb_{idx}.png')



def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open( os.path.join(renders_dir , fname))
        gt = Image.open(os.path.join(gt_dir , fname))
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate_(test_path):

    test_path_list = [os.path.join(test_path, "test", f) for f in sorted(os.listdir(os.path.join(test_path, "test"))) if 'ours' in f]
    test_path = test_path_list[-1]
    print(f"Selected test model is: {test_path}")

    dssim2s = []
    dssim1s = []
    psnrs = []
    lpipss = []
    gt_dir = os.path.join(test_path, "gt")
    renders_dir =os.path.join(test_path , "renders")



    renders, gts, image_names = readImages(renders_dir, gt_dir)
    csvfile = open(os.path.join(test_path, 'eval_ldr.csv'),"w") 
    writer = csv.writer(csvfile)



    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):

        dssim1 =  (1 - ssim(renders[idx], gts[idx])[0]) / 2
        dssim2  =  (1-  msssim(renders[idx], gts[idx])) / 2
        psnr_ = psnr(renders[idx], gts[idx])
        lpips_ =  psnr_ * 0 #  lpips(renders[idx], gts[idx], net_type='alex')

        dssim1s.append(dssim1)
        dssim2s.append(dssim2)
        psnrs.append(psnr_)
        lpipss.append(lpips_)  # this used vgg model


        writer.writerow([ idx , psnr_.item(), dssim1.item(), dssim2.item(), lpips_.item()])
    
    avg_ssim = torch.tensor(dssim1s).mean().item()
    avg_msssim = torch.tensor(dssim2s).mean().item()
    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_lpips = torch.tensor(lpipss).mean().item()
    print(" Avg PSNR : {:>12.7f}".format(avg_psnr, ".5"))
    print(" Avg DSSIM1 : {:>12.7f}".format(avg_ssim, ".5"))
    print(" Avg DSSIM2 : {:>12.7f}".format(avg_msssim, ".5"))
    print(" Avg LPIPS: {:>12.7f}".format(avg_lpips, ".5"))
    print("")

    writer.writerow([ "avg" , avg_psnr, avg_ssim, avg_msssim, avg_lpips])



if __name__ == "__main__":
    # device = torch.device("cuda:2")
    # torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    Mlp = ModelParams(parser)
    
    args = parser.parse_args(sys.argv[1:])
    evaluate_(args.model_path)

