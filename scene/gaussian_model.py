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
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.embedding import Embedding
from scene.spacetime_hash import SpaceTimeHashingField
from scipy.spatial import KDTree
from scene.spacetime_hexplane import HexPlaneField
from scene.regulation import compute_plane_smoothness


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, args, opt,
                 feat_dim: int=32, 
                 n_offsets: int=5, 
                 voxel_size: float=0.01,
                 update_depth: int=3, 
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 ratio : int = 1,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 ):

        self.feat_dim = feat_dim  # 64   anchor 的特征维度
        self.n_offsets = n_offsets  # 10
        self.voxel_size = voxel_size
        self.update_depth = update_depth  # 3
        self.update_init_factor = update_init_factor  # 16
        self.update_hierachy_factor = update_hierachy_factor   # 4
        self.use_feat_bank = use_feat_bank  # false

        self._xyz_bound_min = None
        self._xyz_bound_max = None
        self.appearance_dim = appearance_dim  # 32 这是啥
        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist  # false
        self.add_cov_dist = add_cov_dist  # false
        self.add_color_dist = add_color_dist  # false

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        
        self.opacity_accum = torch.empty(0)
        self.opacity_max = torch.empty(0)  # 动态使用
        self.grad_max = torch.empty(0)  # 动态使用
        self.grad_max_points = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)
        self.dynamic_mask = torch.empty(0)
        self.anchor_demon = torch.empty(0)
        
        self.optimizer = None

        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.hashmap_size = opt.hashmap_size
        self.activation = opt.activation
        self.n_levels = opt.n_levels 
        self.n_features_per_level= opt.n_features_per_level
        self.base_resolution = opt.base_resolution 
        self.n_neurons = opt.n_neurons 
        self.optimizer_mlp_dy = None
        self.hash = opt.hash

        if not self.hash:
            self.hexplane=  HexPlaneField(args.bounds, args.kplanes_config, args.multires, feat_dim ).to("cuda") 

        if self.use_feat_bank:  # false
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, 2*feat_dim),
                nn.ReLU(True),
                nn.Linear( 2*feat_dim, 3),
                nn.Softmax(dim=1)
        ).cuda()

        self.time_embedding = torch.nn.Embedding(300, 32).cuda()  # 时间加上个 embedding 

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            # nn.Linear(feat_dim+3+self.opacity_dist_dim, feat_dim),  # 原版
            nn.Linear( feat_dim , opt.opacity_factor * feat_dim ),
            nn.ReLU(True),
            nn.Linear( opt.opacity_factor * feat_dim, n_offsets),
            # nn.Tanh() # Tanh 会导致一部分 神经高斯无法获得梯度，永久失活
            nn.Sigmoid()
        ).cuda()

        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            # nn.Linear(feat_dim+3+self.cov_dist_dim, feat_dim),  # [35,32]
            nn.Linear(feat_dim, opt.cov_factor * feat_dim),
            nn.ReLU(True),
            nn.Linear(opt.cov_factor * feat_dim, 7 * self.n_offsets),  # [32,70]
        ).cuda()


        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.mlp_color = nn.Sequential(
            # nn.Linear(feat_dim + 3 + self.color_dist_dim + self.appearance_dim, feat_dim), # [65,32]
            nn.Linear(feat_dim  , opt.color_factor * feat_dim), # [65,32]                            去掉这个
            # nn.Linear(feat_dim , opt.color_factor * feat_dim),
            nn.ReLU(True),
            nn.Linear( opt.color_factor * feat_dim, 3 * self.n_offsets),  # [32,30]
            nn.Sigmoid()
        ).cuda()



        self.mlp_offset = nn.Sequential(  # only direction  这个是不是可以提高一点
            nn.Linear( feat_dim  , opt.offset_factor * feat_dim), # [64,32]
            nn.ReLU(True),
            nn.Linear( opt.offset_factor * feat_dim, 3 * self.n_offsets),  # [32,30]
        ).cuda()



# ********************************  dynamic anchor
        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp_opacity_dy = nn.Sequential(
            # nn.Linear(feat_dim+3+self.opacity_dist_dim, feat_dim),  # 原版
            nn.Linear( feat_dim , opt.opacity_factor * feat_dim ),
            nn.ReLU(True),
            # nn.Linear(  opt.opacity_factor * feat_dim , opt.opacity_factor * feat_dim ),
            # nn.ReLU(True),
            nn.Linear( opt.opacity_factor * feat_dim, n_offsets),
            # nn.Tanh() # Tanh 做激活函数, 非常重要
            #   nn.Linear( opt.opacity_factor * feat_dim, 1),
        ).cuda()

        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov_dy = nn.Sequential(
            # nn.Linear(feat_dim+3+self.cov_dist_dim, feat_dim),  # [35,32]
            nn.Linear(feat_dim, opt.cov_factor * feat_dim),
            nn.ReLU(True),
            # nn.Linear(opt.cov_factor * feat_dim, opt.cov_factor * feat_dim),
            # nn.ReLU(True),
            nn.Linear(opt.cov_factor * feat_dim, 7*self.n_offsets),  # [32,70]
            # nn.Linear( opt.cov_factor * feat_dim, 7 ),  # [32,70]
        ).cuda()

        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.mlp_color_dy = nn.Sequential(
            # nn.Linear(feat_dim + 3 + self.color_dist_dim + self.appearance_dim, feat_dim), # [65,32]
            nn.Linear(feat_dim + 3 , opt.color_factor * feat_dim), # [65,32]                            去掉这个
            nn.ReLU(True),
            # nn.Linear(opt.color_factor * feat_dim , opt.color_factor * feat_dim), # [65,32]                            去掉这个
            # nn.ReLU(True),
            nn.Linear( opt.color_factor * feat_dim, 3 * self.n_offsets),  # [32,30]
            # nn.Linear( opt.color_factor * feat_dim, 3 ),  # [32,30]
            # nn.Sigmoid()
        ).cuda()

        self.mlp_offset_dy = nn.Sequential(  # only direction  这个是不是可以提高一点
            nn.Linear( feat_dim  , opt.offset_factor * feat_dim), # [64,32]
            nn.ReLU(True),
            #  nn.Linear(  opt.offset_factor * feat_dim  , opt.offset_factor * feat_dim), # [64,32]
            # nn.ReLU(True),
            nn.Linear( opt.offset_factor * feat_dim, 3 * self.n_offsets),  # [32,30]
            # nn.Linear( opt.offset_factor * feat_dim, 3),  # [32,30]
        ).cuda()


# ********************************  dynamic anchor


    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()

    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._local,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._anchor, 
        self._offset,
        self._local,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity
    
    @property
    def get_cov_mlp(self):
        return self.mlp_cov


    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_offset_mlp(self):
        return self.mlp_offset
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    


    
    #  ******************* dynamic 
    @property
    def get_dy_opacity_mlp(self):
        return self.mlp_opacity_dy
    
    @property
    def get_dy_cov_mlp(self):
        return self.mlp_cov_dy

    @property
    def get_dy_color_mlp(self):
        return self.mlp_color_dy
    
    @property
    def get_dy_offset_mlp(self):
        return self.mlp_offset_dy
    
    #  ******************* dynamic 




    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def voxelize_sample(self, data=None, voxel_size=0.01):  # 体素化
        np.random.shuffle(data)  # 避免数据中存在的顺序性对结果产生影响。
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size  # 去除重复的点，这样每个体素中只保留一个代表点。
        
        return data

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, maxtime):
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points[::self.ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')
        
        
        points = self.voxelize_sample(points, voxel_size= self.voxel_size)  # 
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda() # [N,10,3]
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()  # [N,32]
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        mask = dist2 < 10
        scales = torch.log(torch.sqrt(dist2[mask]))[...,None].repeat(1, 6)  # 感觉可以每个神经高斯都提高一个 scaling
        

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")) # [N,1]

        self._anchor = nn.Parameter(fused_point_cloud[mask].requires_grad_(True))
        self._offset = nn.Parameter(offsets[mask].requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat[mask].requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots[mask].requires_grad_(False))
        self._opacity = nn.Parameter(opacities[mask].requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")
    

        self.maxz, self.minz = torch.amax(self._anchor[:,2]), torch.amin(self._anchor[:,2]) 
        self.maxy, self.miny = torch.amax(self._anchor[:,1]), torch.amin(self._anchor[:,1]) 
        self.maxx, self.minx = torch.amax(self._anchor[:,0]), torch.amin(self._anchor[:,0]) 
        self.maxz = min((self.maxz, 200.0)) # some outliers in the n4d datasets.. 

        if not self.hash:
            self.hexplane = self.hexplane.to("cuda") 

    def get_xyz_bound(self, percentile = 100.0):
        with torch.no_grad():
            if self._xyz_bound_max == None:
                half_percentile = (100 - percentile) / 200
                self._xyz_bound_min = torch.quantile(self._anchor , half_percentile,dim=0)  # 86.6 % ?
                self._xyz_bound_max = torch.quantile(self._anchor , 1 - half_percentile,dim=0)
            return self._xyz_bound_min, self._xyz_bound_max
        

    def init_all(self):
        


        anchors_feat = torch.zeros((self._anchor.shape[0], self.feat_dim)).float().cuda()  # [N,32]
        dist2 = torch.clamp_min(distCUDA2(self._anchor).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        rots = torch.zeros((self._anchor.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((self._anchor.shape[0], 1), dtype=torch.float, device="cuda")) # [N,1]
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")


        self.mlp_opacity = nn.Sequential(
            # nn.Linear(feat_dim+3+self.opacity_dist_dim, feat_dim),  # 原版
            nn.Linear( self.feat_dim , 2 * self.feat_dim ),
            nn.ReLU(True),
            nn.Linear( 2* self.feat_dim, self.n_offsets),
            nn.Tanh() # Tanh 做激活函数, 非常重要
        ).cuda()

        self.mlp_cov = nn.Sequential(
            # nn.Linear(feat_dim+3+self.cov_dist_dim, feat_dim),  # [35,32]
            nn.Linear(self.feat_dim, 2* self.feat_dim),
            nn.ReLU(True),
            nn.Linear(2 * self.feat_dim, 7*self.n_offsets),  # [32,70]
        ).cuda()


        self.mlp_color = nn.Sequential(
            # nn.Linear(feat_dim + 3 + self.color_dist_dim + self.appearance_dim, feat_dim), # [65,32]
            nn.Linear(self.feat_dim + 3 , 2 * self.feat_dim), # [65,32]                            去掉这个
            # nn.Linear(feat_dim , opt.color_factor * feat_dim),
            nn.ReLU(True),
            nn.Linear( 2 * self.feat_dim, 3 * self.n_offsets),  # [32,30]
            # nn.Sigmoid()
        ).cuda()


        self.mlp_offset = nn.Sequential(  # only direction  这个是不是可以提高一点
            nn.Linear( self.feat_dim  , 4 * self.feat_dim), # [64,32]
            nn.ReLU(True),
            nn.Linear(4 * self.feat_dim, 3 * self.n_offsets),  # [32,30]
        ).cuda()



    #  ***************** dynamic 
    def get_dy_xyz_bound(self, percentile = 100.0):
        with torch.no_grad():
            half_percentile = (100 - percentile) / 200
            self._xyz_bound_min = torch.quantile(self._anchor[self.dynamic_mask] , half_percentile,dim=0)  # 86.6 % ?
            self._xyz_bound_max = torch.quantile(self._anchor[self.dynamic_mask] , 1 - half_percentile,dim=0)
            return self._xyz_bound_min, self._xyz_bound_max
    #  ***************** dynamic 

    def training_setup(self, training_args, stage):
        self.percent_dense = training_args.percent_dense # 0.01

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")  # [N,1]
        self.opacity_max = torch.zeros((self.get_anchor.shape[0]  , 1), device="cuda")  # [N,1]
        self.grad_max = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 1), device="cuda")  # [N,1]
        self.grad_max_points = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 3), device="cuda")  # [N,1]


        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")  # [N*10,1]
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda") # [N*10,1]
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda") # [N,1]
        
        self.dynamic_mask = torch.ones((self.get_anchor.shape[0]), device="cuda", dtype=torch.bool)
    


        l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.time_embedding.parameters(), 'lr': 0.002 , "name": "time_embedding"},

                {'params': self.mlp_offset.parameters(), 'lr': training_args.mlp_offset_lr_init, "name": "mlp_offset"},
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)



        if self.hash :
            self.dynamic_module = SpaceTimeHashingField ( self.get_xyz_bound()[0] , self.get_xyz_bound()[1] , training_args.hashmap_size,
            training_args.activation , training_args.n_levels , training_args.n_features_per_level, training_args.base_resolution ,
            training_args.n_neurons , self.feat_dim)
            self.dy_optimizer = torch.optim.Adam (  list(self.dynamic_module.get_params()) , lr = 0.0  )  
            self.dy_scheduler_args = get_expon_lr_func( lr_init = training_args.hash_init_lr * self.spatial_lr_scale ,
                                                        lr_final = training_args.hash_final_lr * self.spatial_lr_scale ,
                                                        lr_delay_mult = 0.01,
                                                        max_steps = training_args.position_lr_max_steps )
        
        else :
            self.hexplane.set_aabb(self.get_xyz_bound()[1],self.get_xyz_bound()[0] )
            Hexplane = [
            {'params': list(self.hexplane.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': list(self.hexplane.get_grid_parameters()), 'lr': training_args.grid_lr_init * self.spatial_lr_scale, "name": "grid"},
            ]
            self.dy_optimizer = torch.optim.Adam(Hexplane, lr=0.0, eps=1e-15)
            self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final= training_args.deformation_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps )    
        
            self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.grid_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)  
            

        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        
        self.mlp_offset_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_offset_lr_init,
                                                    lr_final=training_args.mlp_offset_lr_final,
                                                    lr_delay_mult=training_args.mlp_offset_lr_delay_mult,
                                                    max_steps=training_args.mlp_offset_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)


    def training_setup_fine(self, training_args):

        self.percent_dense = training_args.percent_dense # 0.01
        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")  # [N,1]
        self.opacity_max = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")  # [N,1]
        self.grad_max = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 1), device="cuda")  # [N,1]
        self.grad_max_points = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 3), device="cuda")  # [N,1]

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")  # [N*10,1]
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda") # [N*10,1]
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda") # [N,1]

        l = [
                {'params': self.mlp_offset_dy.parameters(), 'lr': training_args.mlp_offset_lr_init, "name": "mlp_offset"},
                {'params': self.mlp_opacity_dy.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov_dy.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color_dy.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            ]
        
        self.optimizer_mlp_dy = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.dynamic_module = SpaceTimeHashingField ( self.get_xyz_bound()[0] , self.get_xyz_bound()[1] , training_args.hashmap_size,
        training_args.activation , training_args.n_levels , training_args.n_features_per_level, training_args.base_resolution ,
        training_args.n_neurons , self.feat_dim)
        self.dy_optimizer = torch.optim.Adam (  list(self.dynamic_module.get_params()) , lr = 0.0  )  




    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration) 
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_offset":
                lr = self.mlp_offset_scheduler_args(iteration)
                param_group['lr'] = lr

            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration) 
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr

        if self.dy_optimizer !=None:
            for param_group in self.dy_optimizer.param_groups:
                try:
                    if  "grid" in param_group["name"]:
                        lr = self.grid_scheduler_args(iteration)
                        param_group['lr'] = lr
                        # return lr
                    elif param_group["name"] == "deformation":
                        lr = self.deformation_scheduler_args(iteration)
                        param_group['lr'] = lr
                except:
                    lr = self.dy_scheduler_args(iteration)
                    param_group['lr'] = lr
        
        if self.optimizer_mlp_dy != None:
            for param_group in self.optimizer_mlp_dy.param_groups:
                if param_group["name"] == "mlp_opacity":
                    lr = self.mlp_opacity_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "mlp_cov":
                    lr = self.mlp_cov_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "mlp_color":
                    lr = self.mlp_color_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "mlp_offset":
                    lr = self.mlp_offset_scheduler_args(iteration)
                    param_group['lr'] = lr

            
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('dynamic_mask')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        dynamic_mask = self.dynamic_mask.unsqueeze(1).detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation, dynamic_mask), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)






    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        dynamic_mask_name = [p.name for p in plydata.elements[0].properties if p.name.startswith("dynamic_mask")]
        dynamic_mask = np.zeros((anchor.shape[0]))
        for idx, attr_name in enumerate(dynamic_mask_name):
            dynamic_mask = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.dynamic_mask = torch.tensor(dynamic_mask, dtype=torch.bool, device="cuda")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    


        
    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
            
        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]


    def training_statis(self, neural_points, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):

        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()  # [N,10]
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)  # anchor 将其10个神经高斯的透明度累计
        # self.opacity_max[anchor_visible_mask] = torch.max(self.opacity_max[anchor_visible_mask], temp_opacity.sum(dim=1, keepdim=True))
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1) # [10*N]
        combined_mask[anchor_visible_mask] = offset_selection_mask  # anchor 相机可见的情况下，neural point 也可见 opacity大于0 的mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter   # anchor 可见的情况下，neural point 也可见，radii大于0的mask
        
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm   #   [N*10,1]
        self.offset_denom[combined_mask] += 1   #   [N*10,1]



        # 最大梯度的 neural gs
        grad_mask = (self.grad_max[combined_mask] < grad_norm).squeeze(1)
        temp_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1) # [10*N]
        temp_mask[combined_mask] = grad_mask

        self.grad_max[temp_mask] =  grad_norm[grad_mask] 
        self.grad_max_points[temp_mask] = neural_points[update_filter][grad_mask] 


    def anchor_growing(self, grads, threshold, offset_mask, stage="coarse"):  # threshold = 0.0002
        ## 
        init_length = self.get_anchor.shape[0]*self.n_offsets

        for i in range(self.update_depth):  # 划分三层
            # update threshold
            cur_threshold = threshold * ((self.update_hierachy_factor//2)**i)  # update_hierachy_factor = 4
            # mask from grad threshold
            candidate_mask = ( grads >= cur_threshold )
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)  # grid 大于某个阈值，并且可见次数也大于某个阈值
            
            # random pick  ，均值为0、方差为1的正态分布随机数
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))  # 随机选择一些点不进行增长
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length  #  增长的长度
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0) # 增加 candidate


            # if stage == "coarse":
            # all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)  # [N,10,3] 这个有问题
            # else:
            
            all_xyz = self.grad_max_points 
            
            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = 100 * self.voxel_size*size_factor
            grid_coords = torch.round(self.get_anchor / cur_size).int() # anchor 的所处grid

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask] # 将 神经高斯作为新的anchor点
            # if selected_xyz.shape[0] > all_xyz.shape[0] * 0.05:
            #     selected_xyz = selected_xyz[:int(all_xyz.shape[0] * 0.05)]

            selected_grid_coords = torch.round(selected_xyz / cur_size).int() # neural point 所处grid，

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)  # 去除掉坐标重合的神经高斯


            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else: #  grid_coords).all(-1) 第三维所有元素都为true， any(-1) 第维度只有有一个true， 
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1) # [N]
            # 判断 selected_grid_coords_unique 是否有和 grid_coords 相同的点，  去除掉和原本anchor重合的 神经高斯的坐标
            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size  # 不在任意anchor范围内的点，可以用来生成新的anchor

            
            if candidate_anchor.shape[0] > 0:  # 添加anchor，将选出来的神经高斯作为新的anchor，生成
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05  [N,6]
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]

                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }
                

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                temp_grad_max_points = torch.cat([self.grad_max_points,torch.zeros([ self.n_offsets * new_opacities.shape[0], 3], device='cuda').float()], dim=0)
                del self.grad_max_points
                self.grad_max_points = temp_grad_max_points

                temp_opacity_max = torch.cat([self.opacity_max, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_max
                self.opacity_max = temp_opacity_max



                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]


    def adjust_anchor(self, iteration, check_interval = 100, success_threshold=0.8, grad_threshold = 0.0002, min_opacity=0.005, stage="coarse"):

        if iteration % check_interval ==0 : # : and self._anchor.shape[0] < 50000:
            # # adding anchors
            grads = self.offset_gradient_accum / self.offset_denom # [ N * k , 1 ]
            grads[grads.isnan()] = 0.0
            # grads_norm = torch.norm(grads, dim=-1) # 只有经常被看到的点才会增长，因为很少看到的区域往往梯度很大，但是并不需要加点
            grads_norm =  self.grad_max.squeeze(1)  # defsify
            offset_mask = ( self.offset_denom > check_interval * success_threshold * 0.5).squeeze(dim=1)  # 每个neural点必须可见次数大于某个阈值
            offset_mask = torch.logical_and(self.dynamic_mask.repeat(self.n_offsets), offset_mask)   # 动态点才需要

            self.anchor_growing(grads_norm, grad_threshold, offset_mask, stage)

            # update offset_denom
            self.offset_denom[offset_mask] = 0  # 之前的清 0，增加新加入的anchor的offset
            padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                            dtype=torch.int32, device=self.offset_denom.device)
            self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

            self.offset_gradient_accum[offset_mask] = 0
            padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                            dtype=torch.int32, device=self.offset_gradient_accum.device)
            self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

            
            self.grad_max[offset_mask] = 0
            padding_grad_max = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.grad_max.shape[0], 1],
                                            dtype=torch.float32, device=self.offset_denom.device)
            self.grad_max = torch.cat([self.grad_max, padding_grad_max], dim=0)




            padding_dynamic_mask = torch.ones([self.get_anchor.shape[0]- self.dynamic_mask.shape[0]],  # 复制的动态点
                                            dtype=torch.bool, device=self.offset_denom.device)
            self.dynamic_mask = torch.cat([self.dynamic_mask, padding_dynamic_mask], dim=0)


        


        if iteration % (check_interval  * 100000  )== 0:
            # # prune anchors，   如果某个anchor经常出现在视野，但是平均opacity却小于阈值，那么久直接裁剪
            prune_mask = ( self.opacity_accum < min_opacity * self.anchor_demon ).squeeze(dim=1)
            # prune_mask = ( self.opacity_max < 0.05 ).squeeze(dim=1)  # 所有时刻都小于 min_opacity 才进行裁剪
            anchors_mask = (self.anchor_demon > check_interval * success_threshold).squeeze(dim=1) # [N, 1]
            prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
            
            # update offset_denom  去除掉一些点
            offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
            offset_denom = offset_denom.view([-1, 1])
            del self.offset_denom
            self.offset_denom = offset_denom

            offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
            offset_gradient_accum = offset_gradient_accum.view([-1, 1])
            del self.offset_gradient_accum
            self.offset_gradient_accum = offset_gradient_accum
            
            temp_grad_max_points = self.grad_max_points.view([-1, self.n_offsets,3])[~prune_mask]
            temp_grad_max_points = temp_grad_max_points.view([-1, 3])
            del self.grad_max_points
            self.grad_max_points = temp_grad_max_points

            temp_grad_max = self.grad_max.view([-1, self.n_offsets])[~prune_mask]
            temp_grad_max = temp_grad_max.view([-1, 1])
            del self.grad_max
            self.grad_max = temp_grad_max


            # update opacity accum 
            if anchors_mask.sum()>0:
                self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
                self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
                # self.grad_max_points[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()

            temp_opacity_accum = self.opacity_accum[~prune_mask]
            del self.opacity_accum
            self.opacity_accum = temp_opacity_accum

            temp_opacity_max = self.opacity_max[~prune_mask]
            del self.opacity_max
            self.opacity_max = temp_opacity_max

            temp_anchor_demon = self.anchor_demon[~prune_mask]
            del self.anchor_demon
            self.anchor_demon = temp_anchor_demon

            temp_dynamic_mask = self.dynamic_mask[~prune_mask]
            del self.dynamic_mask
            self.dynamic_mask = temp_dynamic_mask


            if prune_mask.shape[0] > 0:
                self.prune_anchor(prune_mask)  # 把 anchor 也裁剪掉
            
            self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")






#$   * ************************************************

    def addgaussians(self,  baduvidx, viewpoint_cam, depthmap, gt_image, shuffle=True):
        def pix2ndc(v, S):
            return (v * 2.0 + 1.0) / S - 1.0
        
        ratiaolist = torch.tensor([1.0]) # 0.7 to ratiostart

        depths = depthmap[:, baduvidx[:,0], baduvidx[:,1]]
        depths = depths.permute(1,0) 

        # maxdepth = torch.amax(depths) # not use max depth, use the top 5% depths? avoid to much growng
        # depths = torch.ones_like(depths) * depthmax # use the max local depth for the scene ?

        u = baduvidx[:,0] # hight y
        v = baduvidx[:,1] # weidth  x 
          
        new_xyz = []
        new_scaling = []
        new_rotation = []

        camera2wold = viewpoint_cam.world_view_transform.T.inverse()
        projectinverse = viewpoint_cam.projection_matrix.T.inverse()
        maxx, minx = self.maxx, self.minx  
        

        for zscale in ratiaolist :
            ndcu, ndcv = pix2ndc(u, viewpoint_cam.image_height), pix2ndc(v, viewpoint_cam.image_width)
            # targetPz = depths*zscale # depth in local cameras..
            if shuffle == True:
                randomdepth = torch.rand_like(depths) - 0.5 # -0.5 to 0.5
                targetPz = (depths + depths/10*(randomdepth)) *zscale 
            else:
                targetPz = depths*zscale # depth in local cameras..
            
            ndcu = ndcu.unsqueeze(1)
            ndcv = ndcv.unsqueeze(1)

            ndccamera = torch.cat((ndcv, ndcu,   torch.ones_like(ndcu) * (1.0) , torch.ones_like(ndcu)), 1) # N,4 ...
            localpointuv = ndccamera @ projectinverse.T.cuda() 
            diretioninlocal = localpointuv / localpointuv[:,3:] # ray direction in camera space 
            rate = targetPz / diretioninlocal[:, 2:3] #  
            localpoint = diretioninlocal * rate
            localpoint[:, -1] = 1
            worldpointH = localpoint @ camera2wold.T.cuda()  # myproduct4x4batch(localpoint, camera2wold) # 
            worldpoint = worldpointH / worldpointH[:, 3:] #  

            xyz = worldpoint[:, :3] 


            xmask = torch.logical_and(xyz[:, 0] > minx, xyz[:, 0] < maxx )

            selectedmask = torch.logical_or(xmask, torch.logical_not(xmask))  #torch.logical_and(xmask, ymask)
            new_xyz.append(xyz[selectedmask]) 



        new_xyz = torch.cat(new_xyz, dim=0)
        tree = KDTree(self._anchor.detach().cpu().numpy())
        _, indices = tree.query(new_xyz.detach().cpu().numpy())
        new_xyz = new_xyz[self.dynamic_mask[indices]]   # 动态点附近的才添加
        _, indices = tree.query(new_xyz.detach().cpu().numpy())
        



        grid_coords = torch.round( self.get_anchor / 0.1 ).int() # anchor 的所处grid
        selected_grid_coords = torch.round( new_xyz / 0.1 ).int() # neural point 所处grid，
        selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)  # 去除掉坐标重合的神经高斯

        ## split data for reducing peak memory calling
        use_chunk = True
        if use_chunk:
            chunk_size = 4096
            max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
            remove_duplicates_list = []
            for i in range(max_iters):
                cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                remove_duplicates_list.append(cur_remove_duplicates)
            
            remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
        else: #  grid_coords).all(-1) 第三维所有元素都为true， any(-1) 第维度只有有一个true， 
            remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1) # [N]
        # 判断 selected_grid_coords_unique 是否有和 grid_coords 相同的点，  去除掉和原本anchor重合的 神经高斯的坐标


        remove_duplicates = ~remove_duplicates
        candidate_anchor = selected_grid_coords_unique[remove_duplicates] * 0.1  # 不在任意anchor范围内的点，可以用来生成新的anchor

        
        if candidate_anchor.shape[0] > 0:  # 添加anchor，将选出来的神经高斯作为新的anchor，生成

            new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda() * 0.1 # *0.05  [N,6]
            new_scaling = torch.log(new_scaling)

            new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
            new_rotation[:,0] = 1.0

            new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

            new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[indices]
            new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

            new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

            d = {
                "anchor": candidate_anchor,
                "scaling": new_scaling,
                "rotation": new_rotation,
                "anchor_feat": new_feat,
                "offset": new_offsets,
                "opacity": new_opacities,
            }
            

            temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
            del self.anchor_demon
            self.anchor_demon = temp_anchor_demon

            temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
            del self.opacity_accum
            self.opacity_accum = temp_opacity_accum

            temp_grad_max_points = torch.cat([self.grad_max_points,torch.zeros([ self.n_offsets * new_opacities.shape[0], 3], device='cuda').float()], dim=0)
            del self.grad_max_points
            self.grad_max_points = temp_grad_max_points


            temp_opacity_max = torch.cat([self.opacity_max, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
            del self.opacity_max
            self.opacity_max = temp_opacity_max

            torch.cuda.empty_cache()
            
            optimizable_tensors = self.cat_tensors_to_optimizer(d)
            self._anchor = optimizable_tensors["anchor"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]
            self._anchor_feat = optimizable_tensors["anchor_feat"]
            self._offset = optimizable_tensors["offset"]
            self._opacity = optimizable_tensors["opacity"]



        # update offset_denom
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                        dtype=torch.int32, device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                        dtype=torch.int32, device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        
        padding_grad_max = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.grad_max.shape[0], 1],
                                        dtype=torch.float32, device=self.offset_denom.device)
        self.grad_max = torch.cat([self.grad_max, padding_grad_max], dim=0)


        padding_dynamic_mask = torch.zeros([self.get_anchor.shape[0]- self.dynamic_mask.shape[0]],  # 复制的动态点
                                        dtype=torch.bool, device=self.offset_denom.device)
        self.dynamic_mask = torch.cat([self.dynamic_mask, padding_dynamic_mask], dim=0)



        return new_xyz.shape[0]








# ************************


    def save_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':

            torch.save(self.mlp_opacity.state_dict(),os.path.join(path, "mlp_opacity.pth"))
            torch.save(self.mlp_cov.state_dict(),os.path.join(path, "mlp_cov.pth"))
            torch.save(self.mlp_color.state_dict(),os.path.join(path, "mlp_color.pth"))
            torch.save(self.mlp_offset.state_dict(),os.path.join(path, "mlp_offset.pth"))
            torch.save(self.time_embedding.state_dict(),os.path.join(path, "time_embedding.pth"))

            if self.hash:
                torch.save(self.dynamic_module.state_dict(), os.path.join(path, "FDHash.pth") )
            else:
                torch.save(self.hexplane.state_dict(),os.path.join(path, "hexplane.pth"))
            # torch.save(self.mlp_opacity_dy.state_dict(),os.path.join(path, "mlp_opacity_dy.pth"))
            # torch.save(self.mlp_cov_dy.state_dict(),os.path.join(path, "mlp_cov_dy.pth"))
            # torch.save(self.mlp_color_dy.state_dict(),os.path.join(path, "mlp_color_dy.pth"))
            # torch.save(self.mlp_offset_dy.state_dict(),os.path.join(path, "mlp_offset_dy.pth"))

        elif mode == 'unite':
            if self.use_feat_bank:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'feature_bank_mlp': self.mlp_feature_bank.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            elif self.appearance_dim > 0:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            else:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    }, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_model(self, path):
        if self.hash:
            print("loading FDHash model from exists{}".format(path))
            self.dynamic_module = SpaceTimeHashingField(self.get_xyz_bound()[0],self.get_xyz_bound()[1], self.hashmap_size , self.activation ,
        self.n_levels ,  self.n_features_per_level,  self.base_resolution , self.n_neurons , self.feat_dim )
            
            weight_dict = torch.load(os.path.join(path,"FDHash.pth"),map_location="cuda")
            self.dynamic_module.load_state_dict(weight_dict)
        else:
            self.hexplane.set_aabb(self.get_xyz_bound()[1],self.get_xyz_bound()[0] )
            weight_dict = torch.load(os.path.join(path,"hexplane.pth"),map_location="cuda")
            self.hexplane.load_state_dict(weight_dict)


        self.mlp_opacity.load_state_dict(torch.load(os.path.join(path, 'mlp_opacity.pth')))
        self.mlp_cov.load_state_dict(torch.load(os.path.join(path, 'mlp_cov.pth')))
        self.mlp_color.load_state_dict(torch.load(os.path.join(path, 'mlp_color.pth')))
        self.mlp_offset.load_state_dict(torch.load(os.path.join(path, 'mlp_offset.pth')))

        self.time_embedding.load_state_dict(torch.load(os.path.join(path, 'time_embedding.pth')))
        # self.mlp_opacity_dy.load_state_dict(torch.load(os.path.join(path, 'mlp_opacity_dy.pth')))
        # self.mlp_cov_dy.load_state_dict(torch.load(os.path.join(path, 'mlp_cov_dy.pth')))
        # self.mlp_color_dy.load_state_dict(torch.load(os.path.join(path, 'mlp_color_dy.pth')))
        # self.mlp_offset_dy.load_state_dict(torch.load(os.path.join(path, 'mlp_offset_dy.pth')))


    def load_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()


        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError




    def _plane_regulation(self):
        multi_res_grids = self.hexplane.grids
        total = 0
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =  [0,1,3]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    
    def _time_regulation(self):
        multi_res_grids = self.hexplane.grids
        total = 0
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =[2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    def _l1_regulation(self):
        multi_res_grids = self.hexplane.grids

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total
    

    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight):
        return plane_tv_weight * self._plane_regulation() + time_smoothness_weight * self._time_regulation() + l1_time_planes_weight * self._l1_regulation()
    

