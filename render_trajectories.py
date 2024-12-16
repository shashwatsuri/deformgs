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
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, get_all_pos
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import glob 
import matplotlib.pyplot as plt
from colormap import colormap
import seaborn as sns
from utils.external import *

tonumpy = lambda x : x.cpu().numpy()
to8 = lambda x : np.uint8(np.clip(x,0,1)*255) 

def project(means3D_deform,viewpoint_camera):
     # projecting to cam frame for later use in optic flow
    means3D_deform = torch.tensor(means3D_deform,device='cuda',dtype=torch.float32)
    means_deform_h = torch.cat([means3D_deform,torch.ones_like(means3D_deform[:,0:1])],dim=1).T 
    cam_transform = viewpoint_camera.full_proj_transform.to(means_deform_h.device).T

    projections = cam_transform.matmul(means_deform_h)
    projections = projections/projections[3,:]

    projections = projections[:2].T
    H, W = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)

    projections_cam = torch.zeros_like(projections).to(projections.device)
    projections_cam[:,0] = ((projections[:,0] + 1.0) * W - 1.0) * 0.5
    projections_cam[:,1] = ((projections[:,1] + 1.0) * H - 1.0) * 0.5
    return projections_cam


def get_mask(projections=None,gaussian_positions=None,depth=None,cam_center=None,height=800,width=800,depth_threshold=0.2):
    if depth.ndim == 3:
        depth = depth[0]
    depth_threshold = 1.0

    # assert none 
    assert projections is not None
    assert gaussian_positions is not None
    assert depth is not None
    assert cam_center is not None
    
    # get the visible projections
    mask_in_image = (projections[:,0] >= 0) & (projections[:,0] < height) & (projections[:,1] >= 0) & \
            (projections[:,1] < width)
    
    depth_mask = np.ones_like(mask_in_image,dtype=bool)

    return depth_mask , mask_in_image

def find_closest_gauss(gt,gauss):
    # gt : N x 3 : numpy array
    # gauss : M x 3 : numpy array
    # return : N x 1
    # for each gt point, find the closest gauss point
    # return shape N x 1 
    gt = torch.tensor(gt,device='cuda',dtype=torch.float32)
    gauss = torch.tensor(gauss,device='cuda',dtype=torch.float32)
    # gauss = gauss[torch.randperm(gauss.shape[0])[:50000]]
    gt = gt.unsqueeze(0).repeat(gauss.shape[0],1,1)
    gauss = gauss.unsqueeze(1).repeat(1,gt.shape[1],1)
    dists = torch.norm(gt-gauss,dim=-1)
    return torch.argmin(dists,dim=0).cpu().numpy()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background,log_deform=False,args=None,gt=None,force_colors=None):  
    n_gaussians = gaussians._xyz.shape[0]
    all_trajs = None
    gt_idxs = None

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        log_deform_path = None
        render_pkg = render(view, gaussians, pipeline, background,log_deform_path=log_deform_path,no_shadow=args.no_shadow,override_color=force_colors,bounding_box=args.bounding_box)
        if args.no_gt:
            gt = None

        if gt_idxs is None:
            if gt is not None:
                gt_t0 = gt[0]
                gt_idxs = find_closest_gauss(gt_t0,render_pkg["means3D_deform"].cpu().numpy())
                n_gaussians = gt_idxs.shape[0]
            else:
                gt_idxs = np.arange(n_gaussians)
        
        if all_trajs is None:
            all_trajs = render_pkg["means3D_deform"][gt_idxs].unsqueeze(0).cpu().numpy()
        else:
            all_trajs = np.concatenate((all_trajs,render_pkg["means3D_deform"][gt_idxs].unsqueeze(0).cpu().numpy()),axis=0)

    np.save(os.path.join(model_path, name, "ours_{}".format(iteration), 'all_trajs.npy'),all_trajs)



def signal_to_colors(signal,mode='minmax',threshold=None):
    # signal: N_gaussians torch tensor
    # output: N_gaussians x 3 torch tensor
    # normalize signal
    if threshold is not None:
        signal_thresh = signal.clone()
        signal_thresh[signal < threshold] = 1.0
        signal_thresh[signal >= threshold] = 0.0
        signal = signal_thresh
    else:
        if mode == 'minmax':
            signal = (signal - torch.min(signal)) / (torch.max(signal) - torch.min(signal))
        elif mode == 'meanstd':
            signal = (signal - torch.mean(signal)) / torch.std(signal) + 0.5
            # clip
            signal = torch.clamp(signal,0,1)

    signal = signal.unsqueeze(1).repeat(1,3)

    return signal

def compute_isometry(gaussians,k_nearest=5,exp=False):
    all_pos = get_all_pos(gaussians) # N_gaussians x N_times x 3 torch tensor
    t_0_pts = all_pos[:,0].detach().cpu().numpy() # N_gaussians x 3 torch tensor
    o3d_dist_sqrd, o3d_knn_indices = o3d_knn(t_0_pts, k_nearest)
    o3d_knn_dists = np.sqrt(o3d_dist_sqrd)
    o3d_knn_dists = torch.tensor(o3d_knn_dists,device="cuda").flatten()
  
    all_pos = all_pos.permute(1,0,2) # N_times x N_gaussians x 3 torch tensor
    all_gaussians_iso = torch.zeros(all_pos.shape[1],device="cuda")

    # o3d_knn_indices : N_gaussians x k_nearest
    # compute distance to each nearest neighbor in each time step
    for i in range(all_pos.shape[0]):
       knn_points = all_pos[i][o3d_knn_indices]
       knn_points = knn_points.reshape(-1,3)

       means_3D_deform_repeated = all_pos[i].unsqueeze(1).repeat(1,k_nearest,1).reshape(-1,3) # N x 3 
       curr_offsets = knn_points - means_3D_deform_repeated
       knn_dists = torch.linalg.norm(curr_offsets,dim=-1)  

       iso_dists = torch.abs(knn_dists - o3d_knn_dists)

       if exp:
            iso_dists = torch.exp(10*iso_dists)-1.0

       # reshape back to N_gaussians x k_nearest
       iso_dists = iso_dists.reshape(-1,k_nearest) 
       iso_dists = torch.sum(iso_dists,dim=-1)
       all_gaussians_iso += iso_dists
    
    return all_gaussians_iso

def compute_velocities(gaussians):
    all_pos = get_all_pos(gaussians) # N_gaussians x N_times x 3 torch tensor

    # compute average velocity for each gaussian
    velocities = all_pos[:,1:] - all_pos[:,:-1] # N_gaussians x N_times-1 x 3
    velocities = torch.norm(velocities,dim=-1) # N_gaussians x N_times-1
    velocities = torch.sum(velocities,dim=-1) # N_gaussians

    return velocities

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool,log_deform=False,user_args=None):
    
    gt_path = os.path.join(dataset.source_path, "gt.npz")
    gt = None
    if os.path.exists(gt_path):
        gt = np.load(gt_path)['traj']
        print("loaded gt from {}".format(gt_path)) 
        print("loaded gt from {}".format(gt_path)) 
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False,user_args=user_args)

        force_colors = None
        if args.viz_velocities:
            velocities = compute_velocities(gaussians)
            force_colors = signal_to_colors(velocities,threshold=0.1)
        if args.viz_isometry:
            isometry = compute_isometry(gaussians,exp=True)
            force_colors = signal_to_colors(isometry,threshold=15)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, 
                       background,log_deform=log_deform,args=user_args,gt=gt,force_colors=force_colors)
        if not skip_test:
            log_folder = os.path.join(args.model_path, "test", "ours_{}".format(scene.loaded_iter))
            delete_previous_deform_logs(log_folder)
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline,
                       background,log_deform=log_deform,args=user_args,gt=gt,force_colors=force_colors)           
        if not skip_video:
            render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,
                       background,log_deform=log_deform,args=user_args,gt=gt,force_colors=force_colors)
 
def delete_previous_deform_logs(folder):
    npz_files = glob.glob(os.path.join(folder,'log_deform_*.npz'),recursive=True)
    for npz_file in npz_files:
        os.remove(npz_file)
            
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--time_skip",type=int,default=None)
    parser.add_argument("--view_skip",default=None,type=int)
    parser.add_argument("--view_ids",type=int,nargs='+',default=None)

    parser.add_argument("--log_deform", action="store_true")
    parser.add_argument("--three_steps_batch",type=bool,default=False)
    parser.add_argument("--show_flow",action="store_true")
    parser.add_argument("--flow_skip",type=int,default=1)
    parser.add_argument("--no_shadow",action="store_true")
    parser.add_argument("--scale",type=float,default=1.0)
    parser.add_argument("--single_cam_video",action="store_true",help='Only render from the first camera for the video viz')
    parser.add_argument("--tracking_window",type=int,default=None)
    parser.add_argument("--no_gt",action="store_true")
    parser.add_argument("--viz_velocities",action="store_true")
    parser.add_argument("--viz_isometry",action="store_true")
    parser.add_argument("--bounding_box",nargs='+',type=float,default=None)

    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), skip_train=False, skip_test=True, skip_video=True,log_deform=args.log_deform,user_args=args)
    