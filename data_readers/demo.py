import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import cv2
import math
import random
import os.path as osp

from glob import glob


class DemoDataset(data.Dataset):
    def __init__(self, root, intrinsics, res_factor=1., depth_factor=1., depth_limit_bottom=0.,
                 depth_limit_top=float("inf")):
        self.init_seed = None

        self.image_list = []
        self.depth_list = []
        self.res_factor = res_factor
        self.depth_factor = depth_factor
        self.depth_limit_bottom = depth_limit_bottom
        self.depth_limit_top = depth_limit_top
        self.intrinsics = np.array(intrinsics)

        images = sorted(glob(osp.join(root, "color/*")))
        depths = sorted(glob(osp.join(root, "depth/*")))

        frames = list(range(0, len(images) - 1))

        for i in frames:
            self.image_list += [[images[i], images[i + 1]]]
            self.depth_list += [[depths[i], depths[i + 1]]]

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

        # Read data
        image1 = cv2.imread(self.image_list[index][0])  # BGR image
        image2 = cv2.imread(self.image_list[index][1])

        depth1 = cv2.imread(self.depth_list[index][0], cv2.IMREAD_ANYDEPTH)
        depth2 = cv2.imread(self.depth_list[index][1], cv2.IMREAD_ANYDEPTH)
        depth1 = np.array(depth1, dtype="float32") * self.depth_factor
        depth2 = np.array(depth2, dtype="float32") * self.depth_factor

        # From numpy to torch
        image1 = torch.from_numpy(image1).float().permute(2, 0, 1)
        image2 = torch.from_numpy(image2).float().permute(2, 0, 1)
        depth1 = torch.from_numpy(depth1).float()
        depth2 = torch.from_numpy(depth2).float()
        intrinsics = torch.from_numpy(self.intrinsics).float()

        # Reduce resolution by res_factor
        if self.res_factor != 1.:
            h_i, w_i = depth1.shape
            h, w = int(math.ceil(h_i / self.res_factor)), int(math.ceil(w_i / self.res_factor))
            sy, sx = float(h) / float(h_i), float(w) / float(w_i)
            image1 = F.interpolate(image1[None], [h, w], mode='bilinear', align_corners=True)[0]
            image2 = F.interpolate(image2[None], [h, w], mode='bilinear', align_corners=True)[0]
            depth1 = F.interpolate(depth1[None, None], [h, w], mode='bilinear', align_corners=True)[0, 0]
            depth2 = F.interpolate(depth2[None, None], [h, w], mode='bilinear', align_corners=True)[0, 0]
            intrinsics *= torch.as_tensor([sx, sy, sx, sy])

        # Limit bottom and top depth bounds
        depth_mask = (depth1 > self.depth_limit_bottom) * (depth1 < self.depth_limit_top) * \
                     (depth2 > self.depth_limit_bottom) * (depth2 < self.depth_limit_top)

        # Valid pixels mask
        valid_mask = depth_mask.unsqueeze(-1)

        depth_scale_factor = 1

        return image1, image2, depth1, depth2, intrinsics, valid_mask, depth_scale_factor
