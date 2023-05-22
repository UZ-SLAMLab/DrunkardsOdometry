import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import cv2
import math
import random
import os.path as osp
import flowpy

from glob import glob

import drunkards_odometry.projective_ops as pops
from .augmentation import ColorAugmentor
from drunkards_odometry.sampler_ops import bilinear_sampler


class DrunkDataset(data.Dataset):
    def __init__(self, root='/media/david/DiscoDuroLinux/Datasets/drunk_dataset',
                 difficulty_level=None, do_augment=True, res_factor=1, scenes_to_use=None, depth_augmentor=False, mode='train', invert_order_prob=0.0):
        self.init_seed = None
        self.res_factor = res_factor
        self.depth_augmentor = depth_augmentor
        self.mode = mode
        self.invert_order_prob = invert_order_prob

        if do_augment:
            self.augmentor = ColorAugmentor()
        else:
            self.augmentor = None

        self.image_list = []
        self.depth_list = []
        self.flow_list = []
        self.pose_list = []
        self.intrinsics_list = []

        intrinsics = np.array([610.17789714, 915.2668457, 512., 512.])  # 1024x1024

        available_scenes = [i.name.lstrip('0') or '0' for i in os.scandir(root) if i.is_dir()]
        available_scenes = list(map(int, available_scenes))
        if isinstance(scenes_to_use, int):
            scenes_to_use = [scenes_to_use]
        if isinstance(available_scenes, int):
            available_scenes = [available_scenes]
        scenes = list(set(scenes_to_use) & set(available_scenes))

        for scene_i in scenes:
            images = sorted(glob(osp.join(root, "{:05d}".format(scene_i), "level{}/color/*.png".format(difficulty_level))))
            if not len(images):
                images = sorted(glob(osp.join(root, "{:05d}".format(scene_i), "level{}/color/color/*.png".format(difficulty_level))))
            depths = sorted(glob(osp.join(root, "{:05d}".format(scene_i), "level{}/depth/*.png".format(difficulty_level))))
            if not len(depths):
                depths = sorted(glob(osp.join(root, "{:05d}".format(scene_i), "level{}/depth/depth/*.png".format(difficulty_level))))
            flows = sorted(glob(osp.join(root, "{:05d}".format(scene_i), "level{}/optical_flow/*.npz".format(difficulty_level))))
            if not len(flows):
                flows = sorted(glob(osp.join(root, "{:05d}".format(scene_i), "level{}/optical_flow/optical_flow/*.npz".format(difficulty_level))))
            poses = osp.join(root, "{:05d}".format(scene_i), "level{}/pose.txt".format(difficulty_level))
            poses_file = open(poses, 'r')
            poses = poses_file.readlines()

            frames = list(range(0, len(poses) - 1))
            wrong_frames = osp.join(root, "{:05d}".format(scene_i), "level{}".format(difficulty_level),
                                    "wrong_frames.txt")
            if osp.exists(wrong_frames):
                wrong_frames = open(wrong_frames, 'r')
                wrong_frames = wrong_frames.read().splitlines()
                wrong_frames = [i for i in wrong_frames if i != '']
                wrong_frames = list(map(int, wrong_frames))
                frames = (frame for frame in frames if frame not in wrong_frames)

            for i in frames:
                self.intrinsics_list += [intrinsics]
                self.image_list += [[images[i], images[i + 1]]]
                self.pose_list += [[poses[i], poses[i + 1]]]
                self.depth_list += [[depths[i], depths[i + 1]]]
                self.flow_list += [[flows[i], flows[i + 1]]]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        invert_order = np.random.rand() < self.invert_order_prob

        # Read data
        image1 = cv2.imread(self.image_list[index][0])  # BGR image
        image2 = cv2.imread(self.image_list[index][1])
        depth1 = cv2.imread(self.depth_list[index][0], cv2.IMREAD_ANYDEPTH)
        depth2 = cv2.imread(self.depth_list[index][1], cv2.IMREAD_ANYDEPTH)
        depth1 = np.array(depth1, dtype="float32") / (2 ** 16 - 1) * 30  # [m]
        depth2 = np.array(depth2, dtype="float32") / (2 ** 16 - 1) * 30  # [m]
        flow2d = np.load(self.flow_list[index][0])['optical_flow']

        pose1 = self.pose_list[index][0].split()
        pose2 = self.pose_list[index][1].split()
        tx1, ty1, tz1 = float(pose1[1]), float(pose1[2]), float(pose1[3])
        tx2, ty2, tz2 = float(pose2[1]), float(pose2[2]), float(pose2[3])
        qx1, qy1, qz1, qw1 = float(pose1[4]), float(pose1[5]), float(pose1[6]), float(pose1[7])
        qx2, qy2, qz2, qw2 = float(pose2[4]), float(pose2[5]), float(pose2[6]), float(pose2[7])
        pose1 = np.array([tx1, ty1, tz1, qx1, qy1, qz1, qw1], dtype="float32")  # Pose world-to-camera, openCV
        pose2 = np.array([tx2, ty2, tz2, qx2, qy2, qz2, qw2], dtype="float32")

        # From numpy to torch
        image1 = torch.from_numpy(image1).float().permute(2, 0, 1)  # RGB
        image2 = torch.from_numpy(image2).float().permute(2, 0, 1)
        depth1 = torch.from_numpy(depth1).float()
        depth2 = torch.from_numpy(depth2).float()
        flow2d = torch.from_numpy(flow2d).float()
        pose1 = torch.from_numpy(pose1).float()
        pose2 = torch.from_numpy(pose2).float()
        intrinsics = torch.from_numpy(self.intrinsics_list[index]).float()

        # Reduce resolution by res_factor
        if self.res_factor != 1.:
            h, w = int(math.ceil(1920 / self.res_factor)), int(math.ceil(1920 / self.res_factor))
            sy, sx = float(h) / float(1920), float(w) / float(1920)
            image1 = F.interpolate(image1[None], [h, w], mode='bilinear', align_corners=True)[0]
            image2 = F.interpolate(image2[None], [h, w], mode='bilinear', align_corners=True)[0]
            depth1 = F.interpolate(depth1[None, None], [h, w], mode='bilinear', align_corners=True)[0, 0]
            depth2 = F.interpolate(depth2[None, None], [h, w], mode='bilinear', align_corners=True)[0, 0]
            flow2d = flow2d.permute(2, 0, 1)[None]
            flow2d = F.interpolate(flow2d, [h, w], mode='bilinear', align_corners=True)[0]
            flow2d = flow2d.permute(1, 2, 0) * torch.as_tensor([sx, sy])
            intrinsics *= torch.as_tensor([sx, sy, sx, sy])

        depth_mask = (depth1 > 0.1) * (depth1 < 30.0) * (depth2 > 0.1) * (depth2 < 30.0)
        flow2d_mask = torch.sum(flow2d ** 2, dim=-1).sqrt() < 250

        if self.depth_augmentor:
            # Augment depth
            s = 0.1 + 1.8 * np.random.rand()
            pose1[:3] *= s
            pose2[:3] *= s
            depth1 = depth1 * s
            depth2 = depth2 * s
        else:
            s = 1.0

        if invert_order:
            flowz = (1.0 / depth1 - 1.0 / depth2).unsqueeze(-1)
            flow2d = - flowpy.forward_warp(flow2d.cpu().detach().numpy(), flow2d.cpu().detach().numpy())
            flow2d = torch.from_numpy(flow2d).float()
            flowxyz = torch.cat([flow2d, flowz], dim=-1)  # flow2d + inverse depth change
        else:
            flowz = (1.0 / depth2 - 1.0 / depth1).unsqueeze(-1)
            flowxyz = torch.cat([flow2d, flowz], dim=-1)  # flow2d + inverse depth change

        if self.augmentor:
            image1, image2 = self.augmentor(image1, image2)

        # Relative pose to go from pose1 to pose2
        pose_gt = torch.from_numpy(pops.absolut_to_relative_poses(pose1, pose2)).float().to(pose1.device)
        pose_gt_inv = pops.pose_from_matrix_to_quat(torch.inverse(pops.pose_from_quat_to_matrix(pose_gt.unsqueeze(0)))).squeeze()

        # Valid pixels mask
        h, w = depth1.shape[:2]
        y1, x1 = torch.meshgrid(
            torch.arange(h).to(flow2d.device).float(),
            torch.arange(w).to(flow2d.device).float())
        x2 = x1 + flow2d[..., 0]
        y2 = y1 + flow2d[..., 1]
        coord2 = torch.stack([x2, y2], dim=-1)
        _, valid_flow = bilinear_sampler(depth2.unsqueeze(0).unsqueeze(0), coord2.unsqueeze(0), mask=True)

        if self.mode == 'train':
            valid_mask = (depth_mask * flow2d_mask
                          * (flow2d[..., 0] > -120) * (flow2d[..., 0] < 120) * (flow2d[..., 1] > -120) * (
                                      flow2d[..., 1] < 120)
                          * (valid_flow > 0).squeeze()).unsqueeze(-1)
        elif self.mode == 'test':
            valid_mask = depth_mask.unsqueeze(-1)

        if not invert_order:
            return image1, image2, depth1, depth2, pose_gt, intrinsics, flowxyz, valid_mask, s
        else:
            return image2, image1, depth2, depth1, pose_gt_inv, intrinsics, flowxyz, valid_mask, s
