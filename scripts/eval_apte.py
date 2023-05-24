import numpy as np
from lietorch import SE3
import torch
import drunkards_odometry.projective_ops as pops
import statistics
from tqdm import tqdm
from matplotlib import pyplot as plt
import os


# Method's name dictionary: key = method's name of the trajectory .txt file; value = method's label plot
methods = {"edam": "EDaM", "droidslam": "DROID-SLAM", "drunkards-odometry-hamlyn-w-deformation": "Ours w/ deformation", "drunkards-odometry-hamlyn-wo-deformation": "Ours w/o deformation"}

# Test scenes' name and Sim(3) scale factor to align each trajectory to the reference one (here EDaM)
scenes = ["test1", "test17"]
scale_factor_for_scene = {"test1": {"edam": 1.0, "droidslam": 0.3200800801484627,  "drunkards-odometry-hamlyn-w-deformation": 0.013571237526591824, "drunkards-odometry-hamlyn-wo-deformation": 0.014709639381740133},
                          "test17": {"edam": 1.0, "droidslam": 1.781018493311159, "drunkards-odometry-hamlyn-w-deformation": 0.01840149096841502, "drunkards-odometry-hamlyn-wo-deformation": 0.031742604665564916}}

# root path to the folder containing the poses of all methods. It should follow this structure:
# --> root
#     --> scene0
#         --> method0
#             --> pose_est.txt            Forward estimated camera trajectory poses
#             --> pose_est_backward.txt   Backward estimated camera trajectory poses
#         ...
#     ...
root = "/.../evaluations_hamyln"

# pose_est.txt and pose_est_backward.txt structure. One line per pose:
# timestamp tx ty tz qx qy qz qw


for scene in scenes:
    plt.figure()
    plt.xlabel("Loop length k in number of frames")
    plt.ylabel(u'APTE\u2096')

    f_metrics = open(os.path.join(root, scene, "metrics.txt"), 'w')

    for method, method_label in methods.items():
        print("###### Evaluating method {} in scene {} ######".format(method, scene))

        pose_forward = os.path.join(root, scene, method, "pose_est.txt")
        pose_backward = os.path.join(root, scene, method, "pose_est_backward.txt")
        scale_factor = scale_factor_for_scene[scene][method]

        # Read poses
        with open(pose_forward) as f:
            forward_poses_read = f.readlines()
        with open(pose_backward) as f:
            backward_poses_read = f.readlines()

        forward_poses = []
        backward_poses = []

        for i in range(len(forward_poses_read) - 1):
            pose1 = forward_poses_read[i].split()
            pose2 = forward_poses_read[i + 1].split()
            tx1, ty1, tz1 = float(pose1[1]), float(pose1[2]), float(pose1[3])
            tx2, ty2, tz2 = float(pose2[1]), float(pose2[2]), float(pose2[3])
            qx1, qy1, qz1, qw1 = float(pose1[4]), float(pose1[5]), float(pose1[6]), float(pose1[7])
            qx2, qy2, qz2, qw2 = float(pose2[4]), float(pose2[5]), float(pose2[6]), float(pose2[7])
            pose1 = np.array([tx1, ty1, tz1, qx1, qy1, qz1, qw1], dtype="float32")  # Pose world-to-camera, openCV
            pose2 = np.array([tx2, ty2, tz2, qx2, qy2, qz2, qw2], dtype="float32")  # Pose world-to-camera, openCV
            pose1[:3] *= scale_factor
            pose2[:3] *= scale_factor
            pose = torch.from_numpy(pops.absolut_to_relative_poses(pose1, pose2)).float().cuda()
            forward_poses.append(SE3(pose))

        for i in range(len(forward_poses_read) - 1):
            pose1 = backward_poses_read[i].split()
            pose2 = backward_poses_read[i + 1].split()
            tx1, ty1, tz1 = float(pose1[1]), float(pose1[2]), float(pose1[3])
            tx2, ty2, tz2 = float(pose2[1]), float(pose2[2]), float(pose2[3])
            qx1, qy1, qz1, qw1 = float(pose1[4]), float(pose1[5]), float(pose1[6]), float(pose1[7])
            qx2, qy2, qz2, qw2 = float(pose2[4]), float(pose2[5]), float(pose2[6]), float(pose2[7])
            pose1 = np.array([tx1, ty1, tz1, qx1, qy1, qz1, qw1], dtype="float32")  # Pose world-to-camera, openCV
            pose2 = np.array([tx2, ty2, tz2, qx2, qy2, qz2, qw2], dtype="float32")  # Pose world-to-camera, openCV
            pose1[:3] *= scale_factor
            pose2[:3] *= scale_factor
            pose = torch.from_numpy(pops.absolut_to_relative_poses(pose1, pose2)).float().cuda()
            backward_poses.append(SE3(pose))

        backward_poses.reverse()
        apte_k_list = []

        for k in tqdm(range(1, len(forward_poses) + 1, 1)):
            final_pose_k = SE3(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).cuda())

            for i in range(k):
                final_pose_k = forward_poses[i] * final_pose_k

            for i in range(k):
                final_pose_k = backward_poses[i] * final_pose_k

            final_pose_k = final_pose_k.vec()
            apte_k = torch.sum(final_pose_k[:3] ** 2, -1).sqrt()
            apte_k_list.append(apte_k.item())

        apte = statistics.mean(apte_k_list)
        print("APTE of scene {} by method {}: {}".format(scene, method_label, apte))
        f_metrics.writelines("APTE of scene {} by method {}: {}".format(scene, method_label, apte) + '\n')
        plt.plot(apte_k_list, label=method_label)

    f_metrics.close()
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(root, scene, 'apte.pdf'))
    # plt.show()
    plt.close()
