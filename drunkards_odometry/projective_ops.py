import copy

import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as R  # Hamilton quaternion convention. qx, qy, qz, qw
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix, quaternion_invert, euler_angles_to_matrix, matrix_to_euler_angles  # qw, qx, qy, qz


from .sampler_ops import *

MIN_DEPTH = 0.01  # 0.05  # TODO: Poner aqui 0.01 metros? O no son metros?
MAX_DEPTH = 30.0

def project(Xs, intrinsics):
    """ Pinhole camera projection """
    X, Y, Z = Xs.unbind(dim=-1)
    fx, fy, cx, cy = intrinsics[:,None,None].unbind(dim=-1)

    x = fx * (X / Z) + cx
    y = fy * (Y / Z) + cy
    d = 1.0 / Z

    coords = torch.stack([x, y, d], dim=-1)
    return coords

def inv_project(depths, intrinsics):
    """ Pinhole camera inverse-projection """

    ht, wd = depths.shape[-2:]
    
    fx, fy, cx, cy = \
        intrinsics[:,None,None].unbind(dim=-1)

    y, x = torch.meshgrid(
        torch.arange(ht).to(depths.device).float(), 
        torch.arange(wd).to(depths.device).float())

    X = depths * ((x - cx) / fx)
    Y = depths * ((y - cy) / fy)
    Z = depths

    return torch.stack([X, Y, Z], dim=-1)

def projective_transform(Ts, depth, intrinsics):
    """ Project points from I1 to I2 """
    
    X0 = inv_project(depth, intrinsics)
    X1 = Ts * X0
    x1 = project(X1, intrinsics)

    valid = (X0[...,-1] > MIN_DEPTH) & (X1[...,-1] > MIN_DEPTH) & (X0[...,-1] < MAX_DEPTH) & (X1[...,-1] < MAX_DEPTH)
    return x1, valid.float()

def induced_flow(Ts, depth, intrinsics):
    """ Compute 2d and 3d flow fields """

    X0 = inv_project(depth, intrinsics)
    X1 = Ts * X0

    x0 = project(X0, intrinsics)
    x1 = project(X1, intrinsics)

    flow2d = x1 - x0
    flow3d = X1 - X0

    valid = (X0[...,-1] > MIN_DEPTH) & (X1[...,-1] > MIN_DEPTH) & (X0[...,-1] < MAX_DEPTH) & (X1[...,-1] < MAX_DEPTH)
    return flow2d, flow3d, valid.float()


def backproject_flow3d(flow2d, depth0, depth1, intrinsics):
    """ compute 3D flow from 2D flow + depth change """

    ht, wd = flow2d.shape[0:2]

    fx, fy, cx, cy = \
        intrinsics[None].unbind(dim=-1)
    
    y0, x0 = torch.meshgrid(
        torch.arange(ht).to(depth0.device).float(), 
        torch.arange(wd).to(depth0.device).float())

    x1 = x0 + flow2d[...,0]
    y1 = y0 + flow2d[...,1]

    X0 = depth0 * ((x0 - cx) / fx)
    Y0 = depth0 * ((y0 - cy) / fy)
    Z0 = depth0

    X1 = depth1 * ((x1 - cx) / fx)
    Y1 = depth1 * ((y1 - cy) / fy)
    Z1 = depth1

    flow3d = torch.stack([X1-X0, Y1-Y0, Z1-Z0], dim=-1)
    return flow3d


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


def relative_to_absoult_poses(pose1, T_c2_c1):
    """
    Given a intial pose1 in world-to-camera and the relative pose transformation to change from camera1 to 2 a
    camera-to-world transformation:
        T_w_c1 (pose1): traslation, quaternions
        T_c2_c1:  traslation, quaternions

    Return final pose2 transformation in world-to-camera:
        T_w_c2 (pose2): traslation, quaternions
        T_w_c2 = (T_c2_c1 * (T_w_c1)^-1)^-1
    """

    if torch.is_tensor(pose1):
        pose1 = pose1.cpu().detach().numpy()
        T_c2_c1 = T_c2_c1.cpu().detach().numpy()

    tx1, ty1, tz1 = pose1[0], pose1[1], pose1[2]
    tx2_1, ty2_1, tz2_1 = T_c2_c1[0], T_c2_c1[1], T_c2_c1[2]
    qx1, qy1, qz1, qw1 = pose1[3], pose1[4], pose1[5], pose1[6]
    qx2_1, qy2_1, qz2_1, qw2_1 = T_c2_c1[3], T_c2_c1[4], T_c2_c1[5], T_c2_c1[6]
    rot_matrix1 = R.from_quat([qx1, qy1, qz1, qw1]).as_matrix()
    rot_matrix2_1 = R.from_quat([qx2_1, qy2_1, qz2_1, qw2_1]).as_matrix()
    T_w_c1 = np.eye(4)
    T_c2_c1 = np.eye(4)
    T_w_c1[:3, :3] = rot_matrix1
    T_c2_c1[:3, :3] = rot_matrix2_1
    T_w_c1[:3, 3] = np.array([tx1, ty1, tz1])
    T_c2_c1[:3, 3] = np.array([tx2_1, ty2_1, tz2_1])
    T_w_c2_4x4 = np.linalg.inv(np.matmul(T_c2_c1, np.linalg.inv(T_w_c1)))
    R_w_c2 = R.from_matrix(T_w_c2_4x4[:3, :3]).as_quat()
    T_w_c2 = np.append(T_w_c2_4x4[:3, 3], R_w_c2)

    return T_w_c2


def pose_from_quat_to_matrix(pose):
    """ Transform from pose in quaternion to pose in matrix.

    Input: torch.tensor([[BATCH_i], [tx, ty, tz, qx, qy, qz, qw]])
    Output: torch.tensor([[BATCH_i], [rxx, rxy, rxz, tx], [ryx, ryy, ryz, ty], [rzx, rzy, rzz, tz], [0, 0, 0, 1]])

    Input shape: BATCH, 7
    Output shape: BATCH, 4, 4
    """
    if pose.dim() == 1:
        pose = pose.unsqueeze(0)

    batch_size = pose.size(0)
    t = pose[:, :3]
    q = pose[:, 3:]  # qx, qy, qz, qw
    qxyz, qw = q.split([3, 1], dim=-1)
    q = torch.cat((qw, qxyz), -1)  # qw, qx, qy, qz
    rot_matrix = quaternion_to_matrix(q)
    pose = torch.zeros(batch_size, 4, 4, device=pose.device)
    pose[:, :3, :3] = rot_matrix
    pose[:, :3, 3] = t
    pose[:, 3, 3] = torch.ones(batch_size)

    return pose


def pose_from_matrix_to_euler(pose):
    """ Transform from pose in matrix to pose in Euler.

        Input: [[rxx, rxy, rxz, tx], [ryx, ryy, ryz, ty], [rzx, rzy, rzz, tz], [0, 0, 0, 1]]
        Output: [tx, ty, tz, qx, qy, qz]

        Input shape: BATCH, 4, 4
        Output shape: BATCH, 6
        """
    if pose.dim() == 1:
        pose = pose.unsqueeze(0)

    rot_matrix = pose[:, :3, :3]
    q = matrix_to_euler_angles(rot_matrix, "XYZ")
    pose = torch.cat((pose[:, :3, 3], q), -1)

    return pose


def pose_from_euler_to_matrix(pose):
    """ Transform from pose in Euler angles to matrix.

        Input: [tx, ty, tz, qx, qy, qz]
        Output: [[rxx, rxy, rxz, tx], [ryx, ryy, ryz, ty], [rzx, rzy, rzz, tz], [0, 0, 0, 1]]

        Input shape: BATCH, 6
        Output shape: BATCH, 4, 4
        """
    if pose.dim() == 1:
        pose = pose.unsqueeze(0)

    t, rot_euler = pose.split([3, 3], dim=-1)
    q = euler_angles_to_matrix(rot_euler, "XYZ")
    batch_size = pose.size(0)
    q_ = torch.cat((q, t.unsqueeze(-1)), -1)
    x = torch.cat((torch.zeros(batch_size, 1, 3), torch.ones(batch_size, 1, 1)), -1).to(q_.device)
    pose_ = torch.cat((q_, x), 1)
    return pose_


def pose_from_matrix_to_quat(pose):
    """ Transform from pose in matrix to pose in quaternion.

    Input: [[rxx, rxy, rxz, tx], [ryx, ryy, ryz, ty], [rzx, rzy, rzz, tz], [0, 0, 0, 1]]
    Output: [tx, ty, tz, qx, qy, qz, qw]

    Input shape: BATCH, 4, 4
    Output shape: BATCH, 7
    """
    if pose.dim() == 1:
        pose = pose.unsqueeze(0)

    rot_matrix = pose[:, :3, :3]
    t = pose[:, :3, 3]
    q = matrix_to_quaternion(rot_matrix)  # qw, qx, qy, qz
    qw, qxyz = q.split([1, 3], dim=-1)
    q = torch.cat((qxyz, qw), -1)  # qx, qy, qz, qw
    pose = torch.cat((t, q), -1)

    return pose


def invert_pose(pose):
    """ Invert a batch of relative pose transformations in quaternions.

    Input: torch.tensor([BATCH * [tx, ty, tz, qx, qy, qz, qw]])
    Output: torch.tensor([BATCH * [tx, ty, tz, qx, qy, qz, qw]])
    """
    q = pose[:, 3:]  # qx, qy, qz, qw
    qxyz, qw = q.split([3, 1], dim=-1)
    q = torch.cat((qw, qxyz), -1)  # qw, qx, qy, qz
    q = quaternion_invert(q)  # qw, qx, qy, qz
    qw, qxyz = q.split([1, 3], dim=-1)
    q = torch.cat((qxyz, qw), -1)  # qx, qy, qz, qw

    return q


def pose_from_quaternion_to_axis_angle(pose: torch.Tensor) -> torch.Tensor:
    """
    Inspired from PyTorch3D.
    Convert pose with rotations given as quaternions to axis/angle.

    Args:
        pose: pose with quaternions with real part last,
            as tensor of shape (..., 7).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    quaternions = pose[..., 3:]  # qx, qy, qz, qw
    qxyz, qw = quaternions.split([3, 1], dim=-1)
    quaternions = torch.cat((qw, qxyz), -1)  # qw, qx, qy, qz

    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # For x small, sin(x/2) is about x/2 - (x/2)^3/6, so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    axis_angles = quaternions[..., 1:] / sin_half_angles_over_angles
    pose = torch.cat((pose[..., :3], axis_angles), dim=-1)

    return pose