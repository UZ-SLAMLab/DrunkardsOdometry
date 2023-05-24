import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from scipy.spatial.transform import Rotation as R  # Hamilton quaternion convention. qx, qy, qz, qw
import os
import sys
sys.path.append('.')
import drunkards_odometry.projective_ops as pops
from lietorch import SE3


class Logger:
    def __init__(self, name, total_steps=0, save_path=None, log_freq=100):
        self.total_steps = total_steps
        self.log_freq = log_freq
        self.running_loss = {}
        self.running_loss_val = {}

        if not save_path:
            save_path = os.getcwd()

        if not os.path.isdir(os.path.join(save_path, 'runs', name)):
            os.makedirs(os.path.join(save_path, 'runs', name))

        self.writer = SummaryWriter(os.path.join(save_path, 'runs', name))

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / self.log_freq for k in sorted(self.running_loss.keys())]
        training_str = "[{:8d}] ".format(self.total_steps + 1)
        metrics_str = ("{:<36.4f} " * len(metrics_data)).format(*metrics_data)

        print("TRAIN:" + training_str + metrics_str)

        for key in self.running_loss:
            val = self.running_loss[key] / self.log_freq
            self.writer.add_scalar(key, val, self.total_steps)
            self.running_loss[key] = 0.0

    def _print_training_status_val(self):
        metrics_data = [self.running_loss_val[k] for k in sorted(self.running_loss_val.keys())]
        training_str = "[{:8d}] ".format(self.total_steps)
        metrics_str = " " * 8 + ("{:<36.4f} " * len(metrics_data)).format(*metrics_data)

        print("VAL:  " + training_str + metrics_str)

        for key in self.running_loss_val:
            val = self.running_loss_val[key]
            self.writer.add_scalar(key, val, self.total_steps)
            self.running_loss_val[key] = 0.0

    def push(self, metrics):
        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps == 0:
            metrics_names = [k for k in sorted(self.running_loss.keys())]
            print("Mode   " + "Steps  " + ' ' * 3 + ("{:35}  " * len(metrics_names)).format(*metrics_names))

        if self.total_steps % self.log_freq == self.log_freq - 1:
            self._print_training_status()
            self.running_loss = {}

        self.total_steps += 1

        return self.total_steps

    def push_val(self, metrics):
        for key in metrics:
            if key not in self.running_loss_val:
                self.running_loss_val[key] = 0.0

            self.running_loss_val[key] += metrics[key]

        self._print_training_status_val()
        self.running_loss_val = {}


def normalize_image(image):
    image = image[:, [2, 1, 0]]
    # Drunkard's Dataset normalization values
    mean = torch.as_tensor([157.7229, 163.4245, 169.4616], device=image.device) / 255.0
    std = torch.as_tensor([60.3560, 58.4104, 55.9718], device=image.device) / 255.0

    return (image / 255.0).sub_(mean[:, None, None]).div_(std[:, None, None])


def absolut_to_relative_poses(pose1, pose2):
    """
    Given two transformation matrices to change from camera1 and 2 to world coordinate frame (world-to-camera):
        T_w_c1 (pose1): traslation, quaternions
        T_w_c2 (pose2): traslation, quaternions
    Return relative transformation matrix to change from camera frame 1 to 2 a camera-to-world transformation:
        T_c2_c1:  traslation, quaternions
        T_c2_w = T_c2_c1 * T_c1_w -> T_w_c2^-1 = T_c2_c1 * T_w_c1^-1 -> T_c2_c1 = T_w_c2^-1 * T_w_c1
    """
    if torch.is_tensor(pose1):
        pose1 = pose1.cpu().detach().numpy()
        pose2 = pose2.cpu().detach().numpy()

    tx1, ty1, tz1 = pose1[0], pose1[1], pose1[2]
    tx2, ty2, tz2 = pose2[0], pose2[1], pose2[2]
    qx1, qy1, qz1, qw1 = pose1[3], pose1[4], pose1[5], pose1[6]
    qx2, qy2, qz2, qw2 = pose2[3], pose2[4], pose2[5], pose2[6]
    rot_matrix1 = R.from_quat([qx1, qy1, qz1, qw1]).as_matrix()
    rot_matrix2 = R.from_quat([qx2, qy2, qz2, qw2]).as_matrix()
    T_w_c1 = np.eye(4)
    T_w_c2 = np.eye(4)
    T_w_c1[:3, :3] = rot_matrix1
    T_w_c2[:3, :3] = rot_matrix2
    T_w_c1[:3, 3] = np.array([tx1, ty1, tz1])
    T_w_c2[:3, 3] = np.array([tx2, ty2, tz2])

    T_c2_c1_4x4 = np.matmul(np.linalg.inv(T_w_c2), T_w_c1)
    R_c2_c1 = R.from_matrix(T_c2_c1_4x4[:3, :3]).as_quat()
    T_c2_c1 = np.append(T_c2_c1_4x4[:3, 3], R_c2_c1)

    return T_c2_c1


def get_pose_errors(pose, pose_gt):
    """Compute normal and relative pose errors.
    """
    pose_error = SE3(pose).inv() * SE3(pose_gt)
    pose_error_log = pose_error.log()
    pose_tra_error = torch.abs(pose_error_log[:, :3])
    pose_tra_error_ME = pose_tra_error
    pose_tra_error_RMSE = torch.sum(pose_tra_error ** 2, -1).sqrt()
    pose_rot_error = torch.abs(pose_error_log[:, 3:])  # rotation in lie log space [rad/s]
    pose_rot_error_ME = pose_rot_error

    pose_error_axisangle = pops.pose_from_quaternion_to_axis_angle(pose_error.vec() + 1e-8)
    pose_rot_error_axisangle_module = pose_error_axisangle[:, 3:].norm(dim=-1)

    return pose_tra_error_ME.mean(), pose_tra_error_RMSE.mean(), \
           pose_rot_error_ME.mean(), pose_rot_error_axisangle_module.mean()


def get_flow3d_tra_errors(flow3d_est, flow3d_gt, valid_mask, return_mean=False):
    flow3d_tra_error = torch.abs(flow3d_est - flow3d_gt)
    flow3d_tra_error_RMSE = torch.sum(flow3d_tra_error ** 2, -1).sqrt()
    flow3d_tra_error_RMSE = flow3d_tra_error_RMSE.view(-1)[valid_mask.view(-1)]

    if return_mean:
        flow3d_tra_error_RMSE = flow3d_tra_error_RMSE.mean()

    return flow3d_tra_error_RMSE


def L1_Charbonnier_loss(x, y, alpha=0.5, eps=1e-6):
    diff = torch.add(x, -y)
    error = torch.pow(diff * diff + eps, alpha)
    return error
