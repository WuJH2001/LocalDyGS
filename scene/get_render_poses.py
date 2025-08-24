


import numpy as np
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from typing import NamedTuple

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    mask: np.array



def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m




def w2c_to_c2w(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    return C2W

def poses_avg(all_c2w):

    center = all_c2w[:, :3, 3].mean(0)
    vec2 = normalize(all_c2w[:, :3, 2].sum(0))
    up = all_c2w[:, :3, 1].sum(0)
    c2w = viewmatrix(vec2, up, center)  # [3,4]
    return c2w


def render_path_spiral(c2w, up, rads, focal_x, focal_y, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])  # 扩展rads到3D
    # c2w 是一个平均的视角
    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        # 计算螺旋路径中的位置
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), np.sin(theta), np.sin(theta*zrate), 1.]) * rads)
        
        # 计算焦距方向上的偏移
        focal_point = np.array([focal_x, focal_y, -focal_x * focal_y, 1.]) 
        z = normalize(c - np.dot(c2w[:3, :4], focal_point))
        
        # 生成相机的视图矩阵
        render_poses.append(viewmatrix(z, up, c))  # [3,5]

    return render_poses


def get_render_poses(train_cam_infos):
    all_c2w = []
    cam_infos = []
    for i , cam in enumerate(train_cam_infos):
        c2w = w2c_to_c2w(cam.R, cam.T)
        all_c2w.append(c2w)
    
    all_c2w = np.stack(all_c2w, axis=0)

    avg_pose = poses_avg(all_c2w)  # [3,4]
    
    up = normalize(all_c2w[:, :3, 1].sum(0))
    # rads = np.percentile(np.abs(all_c2w[:, :3, 3]), 1, 0)  # 百分位函数

    rads = np.array([0.5,0.5,0.5])

    focal_x = fov2focal(cam.FovX, cam.width)
    focal_y = fov2focal(cam.FovY, cam.height)

    render_poses = render_path_spiral( avg_pose, up, rads, focal_x, focal_y, zrate=.5, rots=2, N=250 )
    
    times = [i/20 for i in range(20)]

    for i , pose in enumerate(render_poses):
        c2w = np.zeros((4, 4))
        c2w[:3, :3] = pose[:3, :3]
        c2w[:3, 3] = pose[:3, 3]
        c2w[3, 3] = 1.0
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]
        T = w2c[:3, 3]
        R = R.transpose()

        cam_infos.append(CameraInfo(uid=i, R=R, T=T, FovY=cam.FovY, FovX=cam.FovX+0.8, image=cam.image, time=times[i%20], mask = None,
                            image_path=None, image_name=None, width=cam.width, height=cam.height))

    return cam_infos




def render_path_spiral_360(c2w, up, rads, focal_x, focal_y, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])  # 扩展rads到3D

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        # 计算螺旋路径中的位置
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), np.sin(theta), np.sin(theta*zrate), 1.]) * rads)
        
        # 计算焦距方向上的偏移
        focal_point = np.array([focal_x, focal_y, -focal_x * focal_y, 1.]) 
        z = normalize(c - np.dot(c2w[:3, :4], focal_point))
        
        # 生成相机的视图矩阵
        render_poses.append(viewmatrix(z, up, c))  # [3,5]

    return render_poses


def get_render_poses_360(train_cam_infos):
    all_c2w = []
    cam_infos = []
    for i , cam in enumerate(train_cam_infos):
        c2w = w2c_to_c2w(cam.R, cam.T)
        all_c2w.append(c2w)
    
    all_c2w = np.stack(all_c2w, axis=0)

    avg_pose = poses_avg(all_c2w)  # [3,4]
    
    up = normalize(all_c2w[:, :3, 1].sum(0))
    # rads = np.percentile(np.abs(all_c2w[:, :3, 3]), 1, 0)  # 百分位函数

    rads = np.array([0.3,0.3,0.3])

    focal_x = fov2focal(cam.FovX, cam.width)
    focal_y = fov2focal(cam.FovY, cam.height)

    render_poses = render_path_spiral_360( avg_pose, up, rads, focal_x, focal_y, zrate=.5, rots=2, N=250 )
    
    times = times = [i/20 for i in range(20)]

    for i , pose in enumerate(render_poses):
        c2w = np.zeros((4, 4))
        c2w[:3, :3] = pose[:3, :3]
        c2w[:3, 3] = pose[:3, 3]
        c2w[3, 3] = 1.0
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]
        T = w2c[:3, 3]
        R = R.transpose()

        cam_infos.append(CameraInfo(uid=i, R=R, T=T, FovY=cam.FovY, FovX=cam.FovX+0.8, image=cam.image, time=times[i%20], mask = None,
                            image_path=None, image_name=None, width=cam.width, height=cam.height))

    return cam_infos


def generateBasketballCameras(poses, width, height, FovX, FovY, cam):
    w2cs = np.linalg.inv(poses)
    cam_infos = []
    times = [i/20 for i in range(20)]
    for idx, w2c in enumerate(w2cs):
        R = w2c[:3, :3].transpose()
        T = w2c[:3, 3]

        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY - 0.5 , FovX=FovX , image=cam.image, time= times[idx%20], mask = None,
                              image_path=None, image_name=None, width=width, height=height)
        cam_infos.append(cam_info)
    return cam_infos



def interpolation_frames(train_cam_infos, frames_start_end):
    train_poses = []
    for cam in train_cam_infos[::20]:
        w2c_init = np.zeros((4, 4))
        w2c_init[:3, :3] = cam.R.transpose()
        w2c_init[:3, 3] = cam.T
        w2c_init[3, 3] = 1.0
        train_poses.append(w2c_init)
    train_poses = np.linalg.inv(np.stack(train_poses, axis=0))

    from scene.virtual_poses import interpolate_virtual_poses_sequential

    virtual_poses = interpolate_virtual_poses_sequential(train_poses)
    # virtual_poses = np.concatenate([virtual_poses, virtual_poses[::-1]], axis=0)
    
    val_cam_infos = generateBasketballCameras(virtual_poses[frames_start_end[0]+148:frames_start_end[1]+148], train_cam_infos[0].width, 
                                              train_cam_infos[0].height, train_cam_infos[0].FovX, train_cam_infos[0].FovY,cam)
    return val_cam_infos