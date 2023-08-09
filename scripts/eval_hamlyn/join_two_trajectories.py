import numpy as np
from lietorch import SE3
import torch
import drunkards_odometry.projective_ops as pops
import os

# root path to the folder containing the poses of all methods. It should follow this structure:
# --> root
#     --> scene0
#         --> method0
#             --> pose_est.txt   Forward estimated camera trajectory poses
#     --> scene0_backward
#             --> pose_est.txt   Backward estimated camera trajectory poses
#         ...
#     ...

root = "/.../evaluations_hamlyn"
scenes = ["test1", "test17"]
methods = ["drunkards-odometry-hamlyn-w-deformation", "drunkards-odometry-hamlyn-wo-deformation", "droidslam", "edam"]

for method in methods:
    for scene in scenes:
        forward_pose_path = os.path.join(root, "{}/{}/pose_est.txt".format(scene, method))
        backward_pose_path = os.path.join(root, "{}_backward/{}/pose_est.txt".format(scene, method))
        output_path = os.path.join(root, "{}/{}/pose_est_total.txt".format(scene, method))

        # Read input poses
        with open(forward_pose_path) as f:
            forward_poses = f.readlines()

        with open(backward_pose_path) as f:
            backward_poses = f.readlines()

        f = open(output_path, 'w')  # overwrite the file

        for i, pose in enumerate(forward_poses):
            f.writelines(pose)

        length = len(forward_poses)

        pose_w_c = forward_poses[-1].split()
        tx1, ty1, tz1 = float(pose_w_c[1]), float(pose_w_c[2]), float(pose_w_c[3])
        qx1, qy1, qz1, qw1 = float(pose_w_c[4]), float(pose_w_c[5]), float(pose_w_c[6]), float(pose_w_c[7])
        pose_w_c = SE3(torch.from_numpy(np.array([tx1, ty1, tz1, qx1, qy1, qz1, qw1], dtype="float32")).float().cuda())  # Pose world-to-camera, openCV

        for i in range(len(forward_poses) - 1):
            timestamp = '{:010d}'.format(length + i)
            pose1 = backward_poses[i].split()
            pose2 = backward_poses[i + 1].split()
            tx1, ty1, tz1 = float(pose1[1]), float(pose1[2]), float(pose1[3])
            tx2, ty2, tz2 = float(pose2[1]), float(pose2[2]), float(pose2[3])
            qx1, qy1, qz1, qw1 = float(pose1[4]), float(pose1[5]), float(pose1[6]), float(pose1[7])
            qx2, qy2, qz2, qw2 = float(pose2[4]), float(pose2[5]), float(pose2[6]), float(pose2[7])
            pose1 = np.array([tx1, ty1, tz1, qx1, qy1, qz1, qw1], dtype="float32")  # Pose world-to-camera, openCV
            pose2 = np.array([tx2, ty2, tz2, qx2, qy2, qz2, qw2], dtype="float32")  # Pose world-to-camera, openCV
            pose = SE3(torch.from_numpy(pops.absolut_to_relative_poses(pose1, pose2)).float().cuda())
            pose_w_c = pose * pose_w_c
            pose_w_c_vec = pose_w_c.vec()
            tx, ty, tz = pose_w_c_vec[0].item(), pose_w_c_vec[1].item(), pose_w_c_vec[2].item()
            qx, qy, qz, qw = pose_w_c_vec[3].item(), pose_w_c_vec[4].item(), pose_w_c_vec[5].item(), pose_w_c_vec[6].item()
            row = timestamp + ' ' + str(tx) + ' ' + str(ty) + ' ' + str(tz) + ' ' + str(qx) + ' ' + str(qy) + ' ' + str(qz) + ' ' + str(qw)
            f.writelines(row + '\n')
        f.close()

