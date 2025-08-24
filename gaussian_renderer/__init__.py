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

import torch
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
import tinycudann as tcnn
import torch.nn.functional as F

def generate_full_temporal_gaussians(viewpoint_camera, pc : GaussianModel,  visible_mask=None, is_training=False,  timestamp = None , opt_thro = 0.0 ):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    
    anchor = pc.get_anchor[visible_mask]  # [N,3]
    if timestamp == None:
        timestamp = torch.tensor(viewpoint_camera.time).to(anchor.device).repeat(anchor.shape[0],1)
    else :
        timestamp = torch.tensor(timestamp).to(anchor.device).repeat(anchor.shape[0],1)
    if pc.hash:
        dy_feat, dy_factor = pc.dynamic_module(anchor, timestamp)
    else:
        dy_feat, dy_factor = pc.hexplane(anchor,timestamp)
    sta_feat = pc._anchor_feat[visible_mask]  # [N,32]          
    feat =  dy_factor * dy_feat + ( 1 - dy_factor ) * sta_feat  # dy_factor 
    
    # feat =   sta_feat
    # get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center.cuda().unsqueeze(0)
    ob_dist = ob_view.norm(dim=1, keepdim=True) 
    ob_view = ob_view / ob_dist


    neural_opacity = pc.get_opacity_mlp(feat)  # [N,32+3]

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = ( neural_opacity > opt_thro )  # 
    mask = mask.view(-1)

    # select opacity                               
    opacity = neural_opacity[mask] 

    # get offset's color   
    # color = pc.get_color_mlp(cat_local_view_wodist)  
    color = pc.get_color_mlp(feat)   
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3]) # [mask]

    # The [:,:3] controls the step size of offset. The [:,3:] serves as the base scale for neural gaussian's shape, which means the cov MLP learn a residual scales.
    scale_rot = pc.get_cov_mlp(feat) 
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]

    # offsets
    # grid_offsets =  pc._offset[visible_mask].view([-1,3])  #  [N,10,3] grid_offsets =  grid_offsets[pc.dynamic_mask[visible_mask]]  
    offsets =  pc.get_offset_mlp(feat).view([-1,3])   # pc._offset[pc.dynamic_mask]  #  [N,10,3]  
    grid_scaling = pc.get_scaling[visible_mask] # [N,6]  grid_scaling = grid_scaling[pc.dynamic_mask[visible_mask]]

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)  
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets  ], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets  = masked.split([6, 3, 3, 7, 3], dim = -1 )

    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])

    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]  
    xyz = repeat_anchor + offsets


    if scaling.shape[0]==0:
        pass

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, 
    else:
        return xyz, color, opacity, scaling, rot
    




def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, opt_thro = 0.0, stage="coarse", scaling_modifier = 1.0, iteration = 30000 , retain_grad=False , render_anchor=False, visible_mask=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training

    # filter 
    if iteration  >= 10000 and iteration <= 20000:
        opt_thro = (iteration / 1000) *0.001 - 0.01
    elif iteration > 20000:
        opt_thro = 0.01

    xyz, color, opacity, scaling, rot, neural_opacity, mask = \
                                generate_full_temporal_gaussians(viewpoint_camera, pc, visible_mask=visible_mask, is_training=is_training, opt_thro=opt_thro)



    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
            # dy_dynamics.retain_grad()
        except:
            pass


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5) 
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=1,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    rendered_sta_image = None
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii  = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "render_sta_image": rendered_sta_image,
                "depth_map":  None,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "neural_points":xyz
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                }





def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=1,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:  # false
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation  # [N,4]

    radii_pure = rasterizer.visible_filter(means3D = means3D,  
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0
