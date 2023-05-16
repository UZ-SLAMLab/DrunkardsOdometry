import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import cv2
import random
import os.path as osp

from glob import glob


class HamlynDataset(data.Dataset):
    def __init__(self, root='/home/david/datasets/hamlyn_for_drunk_paper', scenes_to_use=None):
        self.init_seed = None

        self.image_list = []
        self.depth_list = []
        self.intrinsics_list = []
        self.target_res_list = []

        intrinsics_dict = dict.fromkeys(["test1", "test1_backward"], np.array([765.823689, 765.823689, 212.472778, 205.675282]))
        intrinsics_dict.update(dict.fromkeys(["test17", "test17_backward"], np.array([417.903625, 417.903625, 157.208288, 143.735811])))

        target_res_dict = dict.fromkeys(["test1", "test1_backward"], [384, 512])  # height, width
        target_res_dict.update(dict.fromkeys(["test17", "test17_backward"], [256, 288]))

        for scene_i in [scenes_to_use]:
            images = sorted(glob(osp.join(root, scene_i, "color/*.jpg")))
            depths = sorted(glob(osp.join(root, scene_i, "depth/*.png")))

            frames = list(range(0, len(images) - 1))

            for i in frames:
                self.intrinsics_list += [intrinsics_dict[scene_i]]
                self.image_list += [[images[i], images[i + 1]]]
                self.depth_list += [[depths[i], depths[i + 1]]]
                self.target_res_list += [target_res_dict[scene_i]]

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
        depth1 = np.array(depth1, dtype="float32") / 1000.0  # [m]
        depth2 = np.array(depth2, dtype="float32") / 1000.0  # [m]

        # From numpy to torch
        image1 = torch.from_numpy(image1).float().permute(2, 0, 1)  # RGB
        image2 = torch.from_numpy(image2).float().permute(2, 0, 1)
        depth1 = torch.from_numpy(depth1).float()
        depth2 = torch.from_numpy(depth2).float()
        intrinsics = torch.from_numpy(self.intrinsics_list[index]).float()
        # target_res = self.target_res_list[index]
        # original_res = [image1.shape[1], image1.shape[2]] # todo chequear que la altura y anchura estan bien
        #
        # if target_res[0] != original_res[0] or target_res[1] != original_res[1]: # todo creo que esto no hace falta porque es siempre igual
        #     h, w = target_res[0], target_res[1]
        #     sy, sx = float(h) / float(original_res[0]), float(w) / float(original_res[1])
        #     image1 = F.interpolate(image1[None], [h, w], mode='bilinear', align_corners=True)[0]
        #     image2 = F.interpolate(image2[None], [h, w], mode='bilinear', align_corners=True)[0]
        #     depth1 = F.interpolate(depth1[None, None], [h, w], mode='bilinear', align_corners=True)[0, 0]
        #     depth2 = F.interpolate(depth2[None, None], [h, w], mode='bilinear', align_corners=True)[0, 0]
        #     intrinsics *= torch.as_tensor([sx, sy, sx, sy])

        # Limit upper depth bound to 30 cm
        depth_mask = (depth1 > 0.01) * (depth1 < 0.3) * (depth2 > 0.01) * (depth2 < 0.3)

        # Valid pixels mask
        valid_mask = depth_mask.unsqueeze(-1)

        depth_scale_factor = 1

        return image1, image2, depth1, depth2, intrinsics, valid_mask, depth_scale_factor
