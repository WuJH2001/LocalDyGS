import torch
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


def interpolate_virtual_poses(base_cams, n_poses=60):
    all_poses = []
    for i in range(len(base_cams)-1):
        cam0 = base_cams[i]
        for j in range(i+1, len(base_cams)):
            cam1 = base_cams[j]
            for k in range(n_poses):
                ratio = np.sin(((k / n_poses) - 0.5) * np.pi) * 0.5 + 0.5
                
                rots = Rot.from_matrix(np.stack([cam0.R.transpose(), cam1.R.transpose()]))
                key_times = [0, 1]
                slerp = Slerp(key_times, rots)
                rot = slerp(ratio)
                
                pose = np.diag([1.0, 1.0, 1.0, 1.0])
                pose = pose.astype(np.float32)
                pose[:3, :3] = rot.as_matrix()
                pose[:3, 3] = ((1.0 - ratio) * cam0.T + ratio * cam1.T) # w2c
                
                all_poses.append(pose)
    
    all_poses = np.stack(all_poses, axis=0) # n, 4, 4
    
    return all_poses

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def viewmatrix(lookdir, up, position, subtract_position=False):
    """Construct lookat view matrix."""
    vec2 = normalize((position - lookdir) if subtract_position else lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m

def poses_avg(poses):
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world


def interpolate_virtual_poses2(base_cams, n_poses=60):
    
    all_poses = []
    for i in range(len(base_cams)-1):
        pose0 = base_cams[i]
        for j in range(i+1, len(base_cams)):
            pose1 = base_cams[j]
            for k in range(1, n_poses-1):
                ratio = np.sin(((k / n_poses) - 0.5) * np.pi) * 0.5 + 0.5
                
                pose_0 = np.linalg.inv(pose0)
                pose_1 = np.linalg.inv(pose1)
                rot_0 = pose_0[:3, :3]
                rot_1 = pose_1[:3, :3]
                
                rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
                key_times = [0, 1]
                slerp = Slerp(key_times, rots)
                rot = slerp(ratio)
                
                pose = np.diag([1.0, 1.0, 1.0, 1.0])
                pose = pose.astype(np.float32)
                pose[:3, :3] = rot.as_matrix()
                pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3] # w2c
                pose = np.linalg.inv(pose)  # c2w
                
                all_poses.append(pose)
    
    all_poses = np.stack(all_poses, axis=0) # n, 4, 4
    
    return all_poses
            

def interpolate_virtual_poses3(base_cams, n_poses=60):
    
    avg_pose = poses_avg(base_cams[:, :3, :4])
    avg_pose = np.concatenate([avg_pose, np.zeros_like(avg_pose[:1, :])], axis=0)
    avg_pose[-1, -1] = 1.0
    
    all_poses = []
    for i in range(len(base_cams)):
        pose0 = np.diag([1.0, 1.0, 1.0, 1.0])
        pose0[:3, :4] = base_cams[i][:3, :4]
        for k in range(1, n_poses-1):
            ratio = np.sin(((k / n_poses) - 0.5) * np.pi) * 0.5 + 0.5
            
            pose_0 = np.linalg.inv(pose0)
            pose_1 = np.linalg.inv(avg_pose)
            rot_0 = pose_0[:3, :3]
            rot_1 = pose_1[:3, :3]
            
            rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
            key_times = [0, 1]
            slerp = Slerp(key_times, rots)
            rot = slerp(ratio)
            
            pose = np.diag([1.0, 1.0, 1.0, 1.0])
            pose = pose.astype(np.float32)
            pose[:3, :3] = rot.as_matrix()
            pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3] # w2c
            pose = np.linalg.inv(pose)  # c2w
            
            all_poses.append(pose)
    
    all_poses = np.stack(all_poses, axis=0) # n, 4, 4
    
    return all_poses

def interpolate_virtual_poses4(base_cams, near_fars, n_poses=60):
    
    near_fars = np.array(near_fars)
    
    poses = base_cams

    # Find a reasonable 'focus depth' for this dataset as a weighted average
    # of near and far bounds in disparity space.
    close_depth, inf_depth = near_fars.min() * .9, near_fars.max() * 5.
    dt = .75
    focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), 100, 0)
    radii = np.concatenate([radii, [1.]])

    # Generate random poses.
    random_poses = []
    cam2world = poses_avg(poses)
    up = poses[:, :3, 1].mean(0)
    for _ in range(n_poses):
        t = radii * np.concatenate([2 * np.random.rand(3) - 1., [1,]])
        position = cam2world @ t
        lookat = cam2world @ [0, 0, -focal, 1.]
        z_axis = position - lookat
        random_poses.append(viewmatrix(z_axis, up, position))
    
    return np.stack(random_poses, axis=0)

def get_near_virtual_pose(base_cam, near_far, n_poses=1):
    
    near_fars = np.array(near_far)
    
    poses = base_cam

    # Find a reasonable 'focus depth' for this dataset as a weighted average
    # of near and far bounds in disparity space.
    close_depth, inf_depth = near_fars.min() * .9, near_fars.max() * 2.
    dt = .75
    focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), 100, 0)
    radii = np.concatenate([radii, [1.]])

    # Generate random poses.
    random_poses = []
    cam2world = poses_avg(poses)
    up = poses[:, :3, 1].mean(0)
    for _ in range(n_poses):
        t = radii * np.concatenate([2 * np.random.rand(3) - 1., [1,]])
        position = cam2world @ t
        lookat = cam2world @ [0, 0, -focal, 1.]
        z_axis = position - lookat
        random_poses.append(viewmatrix(z_axis, up, position))
    
    return np.stack(random_poses, axis=0)[0]

def interpolate_virtual_poses_sequential(base_cams, n_poses=10):
    
    all_poses = []
    for i in range(len(base_cams)-1):
        pose0 = base_cams[i]
        pose1 = base_cams[i+1]
        for k in range(n_poses):
            ratio = np.sin(((k / n_poses) - 0.5) * np.pi) * 0.5 + 0.5
            
            pose_0 = np.linalg.inv(pose0)
            pose_1 = np.linalg.inv(pose1)
            rot_0 = pose_0[:3, :3]
            rot_1 = pose_1[:3, :3]
            
            rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
            key_times = [0, 1]
            slerp = Slerp(key_times, rots)
            rot = slerp(ratio)
            
            pose = np.diag([1.0, 1.0, 1.0, 1.0])
            pose = pose.astype(np.float32)
            pose[:3, :3] = rot.as_matrix()
            pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3] # w2c
            pose = np.linalg.inv(pose)  # c2w
            
            all_poses.append(pose)
    
    all_poses = np.stack(all_poses, axis=0) # n, 4, 4
    
    return all_poses