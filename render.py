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
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import imageio
import numpy as np
import torch
from scene import Scene
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args,OptimizationParams ,ModelHiddenParams
from gaussian_renderer import GaussianModel
from gaussian_renderer import prefilter_voxel, render
from time import time
# import torch.multiprocessing as mp
import threading
import concurrent.futures
from PIL import Image



def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
import matplotlib.pyplot as plt
import numpy as np


def render_set(opt,model_path, name, iteration, views, gaussians, pipeline, background, cam_type):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    # breakpoint()
    print("point nums:",gaussians._anchor.shape[0])
    all_time = 0

 
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        time1 = time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)  # 判断是否可见 
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        rendering = render(view, gaussians, pipeline, background, stage="fine", visible_mask = voxel_visible_mask, retain_grad=retain_grad)["render"]
        time2 = time()
        all_time += (time2-time1)

        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)
        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt  = view['image'].cuda()
            gt_list.append(gt)

    # time2=time()
    print("FPS:",(len(views)-1)/all_time)

    print("writing training images.") 
    multithread_write(gt_list, gts_path) 
    print("writing rendering images.") 
    multithread_write(render_list, render_path) 



    
 




def render_sets( opt , hyper, dataset : ModelParams, frames_start_end, iteration : int, pipeline : PipelineParams,  skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(hyper, opt,dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        # iteration = 7000
        scene = Scene(dataset, gaussians, frames_start_end = frames_start_end, load_iteration=iteration, shuffle=False)
        cam_type=scene.dataset_type
        bg_color = [1,1,1]  if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(opt,dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,cam_type)
        if not skip_test:
            render_set(opt,dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,cam_type)
        if not skip_video:
            render_set(opt,dataset.model_path,"video2",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,cam_type)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--frames_start_end", type=int, nargs=2, default=[0, 300], help="Start and end frames")

    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.general_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # enable logging
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(op.extract(args), hp.extract(args),   model.extract(args), args.frames_start_end, args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video)
    