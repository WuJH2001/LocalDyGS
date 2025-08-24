_base_ = './default.py'




OptimizationParams = dict(
    dataloader=True,
    position_lr_init = 0.00016,   # 试试移动 seed point
    position_lr_final = 0.0000016,


    iterations = 40000 ,
    hash_init_lr = 0.0002 ,
    hash_final_lr = 0.000002 ,
    hashmap_size = 17 ,  # 17
    activation = "ReLU",
    n_levels = 16,  # 16 
    n_features_per_level = 8,  #  4
    base_resolution = 16,  #  16 
    n_neurons = 128,
    opacity_factor = 2 ,
    cov_factor = 2 ,
    color_factor = 2 , 
    offset_factor = 4  ,

    start_stat = 1500,
    update_from = 1600,
    update_interval = 100,
    update_until = 15000,
    success_threshold = 1.0,
    densify_grad_threshold = 0.001,
    percentile = 100
)

ModelParams = dict(
    sh_degree = 3,
    feat_dim = 64,
    n_offsets = 10,
    voxel_size =  0.001 ,# if voxel_size<=0, using 1nn dist 是不是太小了，0.1差不多  0.001
    update_depth = 3,
    update_init_factor = 16,
    update_hierachy_factor = 4,

)
