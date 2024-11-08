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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.dataset import FourDGSdataset, MDNerfDataset
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import Dataset

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], load_coarse=False,user_args=None,freeze_gaussians=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}
        
        time_skip = None
        view_skip = None
        view_ids = None
        if user_args is not None:
            time_skip = user_args.time_skip
            view_skip = user_args.view_skip
            view_ids = user_args.view_ids

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            #load json to check if blender or panopto 
            with open(os.path.join(args.source_path, "transforms_train.json")) as f:
                data = json.load(f)
            # check if 'camera_angle_x' or 'fl_x' in header of json file
            # print all keys of json
            if 'camera_angle_x' in data.keys() or 'fl_x' in data.keys():
                print("Found transforms_train.json file with global intrinsics, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval,time_skip=time_skip,view_skip=view_skip,view_ids=view_ids)            
            else:
                print("Found transforms_train.json file without global intrinsics, assuming Panopto data set!")
                scene_info = sceneLoadTypeCallbacks["Panopto"](args.source_path, args.white_background, args.eval,time_skip=time_skip,view_skip=view_skip,scale=user_args.scale)
            
            print("Found transforms_train.json file, assuming Blender data set!")
            
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            scene_info = sceneLoadTypeCallbacks["dynerf"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path,"dataset.json")):
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, False, args.eval)
        elif os.path.exists(os.path.join(args.source_path,"points3D_multipleview.ply")):
            scene_info = sceneLoadTypeCallbacks["MultipleView"](args.source_path)
            dataset_type="MultipleView"
        else:
            assert False, "Could not recognize scene type!"
        self.maxtime = scene_info.maxtime
        
        if not freeze_gaussians:
            gaussians.all_times = scene_info.all_times

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        if user_args.three_steps_batch:
            print("Loading Training Cameras, MDNeRF")
            self.train_camera = MDNerfDataset(scene_info.train_cameras, args)
            
            self.train_camera_t0 = MDNerfDataset(scene_info.train_cameras, args, only_t0=True)

            print("Loading Test Cameras, MDNeRF")
            self.test_camera = MDNerfDataset(scene_info.test_cameras, args)
            
        else:
            print("Loading Training Cameras, 4DGS")
            self.train_camera = FourDGSdataset(scene_info.train_cameras, args)
            print("Loading Test Cameras, 4DGS")
            self.test_camera = FourDGSdataset(scene_info.test_cameras, args)
        self.train_camera_individual = FourDGSdataset(scene_info.train_cameras, args)
        self.test_camera_individual = FourDGSdataset(scene_info.test_cameras, args)
        
        print("Loading Video Cameras")
        self.video_camera = cameraList_from_camInfos(scene_info.video_cameras,-1,args)
        if not freeze_gaussians:
            xyz_max = scene_info.point_cloud.points.max(axis=0)
            xyz_min = scene_info.point_cloud.points.min(axis=0)
            self.gaussians._deformation.deformation_net.grid.set_aabb(xyz_max,xyz_min)
            if self.loaded_iter:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"))
                self.gaussians.load_model(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                    ))
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)

    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)
    def getTrainCameras(self, scale=1.0):
        return self.train_camera
    
    def getTrainCamerasT0(self, scale=1.0):
        return self.train_camera_t0

    def getTestCameras(self, scale=1.0):
        return self.test_camera
    
    def getTrainCamerasIndividual(self, scale=1.0):
        return self.train_camera_individual
    
    def getTestCamerasIndividual(self, scale=1.0):
        return self.test_camera_individual
    
    def getVideoCameras(self, scale=1.0):
        return self.video_camera