import torch
import tinycudann as tcnn
import torch.nn as nn

class SpaceTimeHashingField(torch.nn.Module):
    def __init__(self, xyz_bound_min, xyz_bound_max , hashmap_size=16, activation = "ReLU", n_levels = 16,
        n_features_per_level=4, base_resolution = 16 ,n_neurons = 128 , feat_dim = 64  ):
        super(SpaceTimeHashingField, self).__init__()

        self.feat_dim = feat_dim

        self.enc_model = tcnn.NetworkWithInputEncoding(
            n_input_dims = 4 ,
            n_output_dims = feat_dim ,    # as same as 4dgs
            encoding_config={
                "otype": "HashGrid" ,
                "n_levels": n_levels ,
                "n_features_per_level": n_features_per_level,  # 2
                "log2_hashmap_size": hashmap_size ,
                "base_resolution": base_resolution ,
                "per_level_scale": 2.0 ,
            },

            network_config={
                "otype": "FullyFusedMLP",
                "activation": activation,
                "output_activation": "ReLU",  
                "n_neurons": n_neurons,
                "n_hidden_layers": 1 ,
            },
        )


        self.mlp_mask = tcnn.Network(
            n_input_dims= 4,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": 1,
            },
        )

        self.register_buffer('xyz_bound_min',xyz_bound_min)
        self.register_buffer('xyz_bound_max',xyz_bound_max)

    def dump(self, path):
        torch.save(self.state_dict(),path)
        

    def get_contracted_xyz(self, xyz):  # 远离中心的一些浮点可以不要
        with torch.no_grad():
            contracted_xyz=(xyz-self.xyz_bound_min)/(self.xyz_bound_max-self.xyz_bound_min)
            return contracted_xyz


    def forward(self, xyz:torch.Tensor, time):

        contracted_xyz=self.get_contracted_xyz(xyz)                          # Shape: [N, 3]
        
        mask = ( contracted_xyz >= 0 ) & ( contracted_xyz <= 1 )
        mask = mask.all(dim=1)
        hash_inputs = torch.cat([contracted_xyz[mask], time[mask]],dim=-1)

        dynamic_feature_out =  self.enc_model(hash_inputs) 
        temp_dynamics = self.mlp_mask( hash_inputs )    # 这里是不是也有问题
        # temp_dynamics = self.mlp_mask( dynamic_feature )
        # dynamics =    torch.log(torch.exp(temp_dynamics) + 1) + 0.03  # [0.03, + \infty] torch.nn.Softplus(temp_dynamics) + 0.03  
        # dynmiac_mask = torch.sigmoid(temp_dynamics)

        dynamic_feature =  torch.zeros((xyz.shape[0],self.feat_dim),  device="cuda")
        dynamic_feature[mask] = dynamic_feature_out.float()

        dynmiac_mask =  torch.zeros((xyz.shape[0],1),  device="cuda")
        dynmiac_mask[mask] = torch.sigmoid(temp_dynamics.float())
        
        if torch.isnan(dynmiac_mask).any():
            pass

        return  dynamic_feature, dynmiac_mask
    


    def get_params (self):

        parameter_list = []
        for name, param in self.named_parameters():
            parameter_list.append(param)
        return parameter_list
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters(): # enc_model.para
            if  "enc_model" not in name:
                parameter_list.append(param)
        return parameter_list


    def get_hash_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "enc_model" in name:
                parameter_list.append(param)  
        return parameter_list

        
        