_base_ = './default.py'




OptimizationParams = dict(
    dataloader=True,
    iterations = 30000 ,
    hash_init_lr = 0.0002 ,
    hash_final_lr = 0.000002 ,
    hashmap_size = 17 ,  
    activation = "ReLU",
    n_levels = 16,
    n_features_per_level = 8,
    base_resolution = 16,
    n_neurons = 128,
    opacity_factor = 2 ,
    cov_factor = 2 ,
    color_factor = 2 , 
    offset_factor = 4  ,

    start_stat = 350020000,
    update_from = 3600,
    update_interval = 100,
    update_until = 15000,

    success_threshold = 1.0,
    densify_grad_threshold = 0.001,
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
