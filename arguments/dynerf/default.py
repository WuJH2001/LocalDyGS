
ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2 ,
     'input_coordinate_dim': 4 ,
     'output_coordinate_dim': 16 ,
     'resolution': [64, 64, 64, 150]
    },
    multires = [1,2],
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
    empty_voxel=False,
)

OptimizationParams = dict(
    dataloader=True,
    # hash = False,
    iterations = 40000,
    hash_init_lr = 0.0002,
    hash_final_lr = 0.000002,
    hashmap_size = 16,
    activation = "ReLU",
    n_levels = 16,
    n_features_per_level = 4,
    base_resolution = 16,    # 这个是不是可以调一下
    n_neurons = 128,
    opacity_factor = 2 ,
    cov_factor = 2 ,
    color_factor = 2, 
    offset_factor = 4 ,

    offset_lr_init = 0.01,
    offset_lr_final = 0.0001,

    opacity_lr  = 0.02,
    scaling_lr  = 0.007,
    rotation_lr = 0.002,
            
    mlp_opacity_lr_init = 0.002,
    mlp_opacity_lr_final = 0.00002  ,

    mlp_cov_lr_init = 0.004,
    mlp_cov_lr_final = 0.004,

    mlp_offset_lr_init = 0.008,
    mlp_offset_lr_final = 0.00005,

    mlp_color_lr_init = 0.008,
    mlp_color_lr_final = 0.00005,


    start_stat = 1500,
    update_from = 3000,
    update_interval = 100,
    update_until = 15_000,
    success_threshold = 0.8,
    densify_grad_threshold = 0.0002,
)

ModelParams = dict(
    sh_degree = 3,
    feat_dim = 64,
    n_offsets = 10,
    voxel_size =  0.001 ,# if voxel_size<=0, using 1nn dist
    update_depth = 3,
    update_init_factor = 16,
    update_hierachy_factor = 4,

)
