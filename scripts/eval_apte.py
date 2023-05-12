from scipy.spatial.transform import Rotation
import numpy as np
from scipy.spatial.transform import Rotation as R  # Hamilton quaternion convention. qx, qy, qz, qw
from lietorch import SE3
import torch
import drunkards_odometry.projective_ops as pops
import statistics
import math
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import argparse
from pathlib import Path


# parser = argparse.ArgumentParser()
# parser.add_argument('--pose_forward', type=str)
# parser.add_argument('--pose_backward', type=str)
# parser.add_argument('--save_folder', type=str)
#
# args = parser.parse_args()
#
# pose_forward = args.pose_forward
# pose_backward = args.pose_backward
# save_folder = args.save_folder

# methods = ["edam", "droidslam", "baseline_newGt_lr0.0002_invertImagesProb0.5_trainedInLevel0_epoch3_fr2", "baseline_newGt_lr0.0002_invertImagesProb0.5Mar02_23-45-36_epoch3_fr2", "baseline_newGt_lr0.0002_invertImagesProb0.5Mar02_23-45-36_epoch9_fr2", "baseline_newGt_lr0.0002_invertImagesProb0.5Mar02_23-45-3_trainedInLevel1_epoch9", "baseline_newGt_lr0.0002_invertImagesProb0.5_trainedInLevel1", "baseline_newGt_lr0.0002_invertImagesProb0.5_trainedInLevel0_epoch3"]
methods = ["edam", "droidslam", "baseline_newGt_lr0.0002_invertImagesProb0.5_trainedInLevel1", "baseline_newGt_lr0.0002_invertImagesProb0.5_trainedInLevel0_epoch3"]
scale_factor_for_methods_test1 = {"edam": 1.0, "droidslam": 0.3200800801484627, "baseline_newGt_lr0.0002_invertImagesProb0.5_trainedInLevel0_epoch3_fr2": 0.02984506483094551, "baseline_newGt_lr0.0002_invertImagesProb0.5Mar02_23-45-36_epoch3_fr2": 0.0623944385752784, "baseline_newGt_lr0.0002_invertImagesProb0.5Mar02_23-45-36_epoch9_fr2": 0.03718791133108942, "baseline_newGt_lr0.0002_invertImagesProb0.5Mar02_23-45-3_trainedInLevel1_epoch9": 0.01318436637251356, "baseline_newGt_lr0.0002_invertImagesProb0.5_trainedInLevel1": 0.013571237526591824, "baseline_newGt_lr0.0002_invertImagesProb0.5_trainedInLevel0_epoch3": 0.014709639381740133}
scale_factor_for_methods_test17 = {"edam": 1.0, "droidslam": 1.781018493311159, "baseline_newGt_lr0.0002_invertImagesProb0.5_trainedInLevel0_epoch3_fr2": 0.10337628549001582, "baseline_newGt_lr0.0002_invertImagesProb0.5Mar02_23-45-36_epoch3_fr2": 0.054199268729278105, "baseline_newGt_lr0.0002_invertImagesProb0.5Mar02_23-45-36_epoch9_fr2": 0.07984214281480992, "baseline_newGt_lr0.0002_invertImagesProb0.5Mar02_23-45-3_trainedInLevel1_epoch9": 0.03242012805889243, "baseline_newGt_lr0.0002_invertImagesProb0.5_trainedInLevel1": 0.01840149096841502, "baseline_newGt_lr0.0002_invertImagesProb0.5_trainedInLevel0_epoch3": 0.031742604665564916}
scale_factor_for_methods_test22 = {"edam": 1.0, "droidslam": 3.085550549406003, "baseline_newGt_lr0.0002_invertImagesProb0.5_trainedInLevel0_epoch3_fr2": 0.22010561551782692, "baseline_newGt_lr0.0002_invertImagesProb0.5Mar02_23-45-36_epoch3_fr2": 0.2963121921481687, "baseline_newGt_lr0.0002_invertImagesProb0.5Mar02_23-45-36_epoch9_fr2": 0.1912969340786122, "baseline_newGt_lr0.0002_invertImagesProb0.5Mar02_23-45-3_trainedInLevel1_epoch9": 0.16501048727785395, "baseline_newGt_lr0.0002_invertImagesProb0.5_trainedInLevel1": 0.13100142473497675, "baseline_newGt_lr0.0002_invertImagesProb0.5_trainedInLevel0_epoch3": 0.07317189571772019}
scale_factor_for_scene = {"test1": scale_factor_for_methods_test1, "test17": scale_factor_for_methods_test17, "test22": scale_factor_for_methods_test22}
scenes = ["test1", "test17"]
# scenes = ["test17"]
apte_accumulative = False
show_plots = False
minimum_loop_frames = 1  # 50 100 default 1
save_metrics = False



# methods = ["edam"] #todo comentar
for scene in scenes:
    plt.figure()
    plt.xlabel("Loop length k in number of frames")
    plt.ylabel(u'APTE\u2096')

    for method in methods:
        if method == "baseline_newGt_lr0.0002_invertImagesProb0.5_trainedInLevel1":
            method_label = "Ours w/ deformation"
        elif method == "baseline_newGt_lr0.0002_invertImagesProb0.5_trainedInLevel0_epoch3":
            method_label = "Ours w/o deformation"
        elif method == "edam":
            method_label = "EDaM"
        elif method == "droidslam":
            method_label = "DROID-SLAM"

        pose_forward = "/media/david/DiscoDuroLinux/Datasets/evaluations_drunk/evaluations_hamlyn/{}/{}/stamped_traj_estimate.txt".format(scene, method)
        pose_backward = "/media/david/DiscoDuroLinux/Datasets/evaluations_drunk/evaluations_hamlyn/{}_backward/{}/stamped_traj_estimate.txt".format(scene, method)
        scale_factor = scale_factor_for_scene[scene][method]
        if scale_factor == 1.0:
            if minimum_loop_frames == 1:
                save_folder = "/media/david/DiscoDuroLinux/Datasets/evaluations_drunk/evaluations_hamlyn/{}/{}".format(scene, method)
            else:
                name = method + "_minLoopFrames{}".format(minimum_loop_frames)
                save_folder = "/media/david/DiscoDuroLinux/Datasets/evaluations_drunk/evaluations_hamlyn/{}/{}".format(scene, name)
        else:
            if minimum_loop_frames == 1:
                name = method + "_aligned_with_scale_factor_" + str(float(f'{scale_factor:.5f}')) + "_minLoopFrames{}".format(minimum_loop_frames)
            else:
                name = method + "_aligned_with_scale_factor_" + str(float(f'{scale_factor:.5f}'))
            save_folder = "/media/david/DiscoDuroLinux/Datasets/evaluations_drunk/evaluations_hamlyn/{}/{}".format(scene, name)

        save_image_folder = "/media/david/DiscoDuroLinux/Datasets/evaluations_drunk/evaluations_hamlyn/qualitative2/{}".format(scene)
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        Path(save_image_folder).mkdir(parents=True, exist_ok=True)


        print("###### Evaluating method {} in scene {} ######".format(method, scene))

        # Read input poses
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
            # pose = absolut_to_relative_poses(pose1, pose2)
            # pose = torch.from_numpy(pose).float().cuda()
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
        apre_k_list = []
        forward_trajectory_length_list = []
        trajectory_length_list = []
        apte_k_2_list = []

        for k in tqdm(range(minimum_loop_frames, len(forward_poses) + 1, 1)):
        # for k in [len(forward_poses)]:
            # if k == 10:
            #     break
            final_pose_k = SE3(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).cuda())
            trajectory_length_k = 0.0
            if apte_accumulative:
                pose_k_list = []
                pose_k_list.append(final_pose_k)

            for i in range(k):
                final_pose_k = forward_poses[i] * final_pose_k
                trajectory_length_k += torch.sum(forward_poses[i].vec()[:3] ** 2, -1).sqrt()
                if apte_accumulative:
                    pose_k_list.append(final_pose_k)

            for i in range(k):
                final_pose_k = backward_poses[i] * final_pose_k
                trajectory_length_k += torch.sum(backward_poses[i].vec()[:3] ** 2, -1).sqrt()
                if apte_accumulative:
                    pose_k_list.append(final_pose_k)

            if apte_accumulative:
                apte_k_2 = 0.0
                for j in range(int((len(pose_k_list) - 1) / 2)):
                    error = (pose_k_list[j].inv() * pose_k_list[len(pose_k_list) - j - 1]).vec()
                    error = torch.sum(error[:3] ** 2, -1).sqrt()
                    apte_k_2 += error
                apte_k_2_list.append(apte_k_2)

            final_pose_k = final_pose_k.vec() # hacer el rmse de la traslacion e ir guardando estos apte_k en una lista para poder plotearlos
            apte_k = torch.sum(final_pose_k[:3] ** 2, -1).sqrt()
            final_pose_axisangle = pops.pose_from_quaternion_to_axis_angle(final_pose_k.unsqueeze(0)).squeeze()
            apre_k = final_pose_axisangle[3:].norm(dim=-1) * 180.0 / math.pi  # [º]

            apte_k_list.append(apte_k.item())
            apre_k_list.append(apre_k.item())
            # forward_trajectory_length_list.append(forward_trajectory_length_k)
            trajectory_length_list.append(trajectory_length_k)

        apte_mean = statistics.mean(apte_k_list)
        apte_median = statistics.median(apte_k_list)
        apre_mean = statistics.mean(apre_k_list)
        apre_median = statistics.median(apre_k_list)
        apte_sum = sum(apte_k_list)
        apte_max = max(apte_k_list)
        apre_sum = sum(apre_k_list)

        apte_k_list_rel_mean = list(map(lambda x: x / apte_mean, apte_k_list))
        apte_k_list_rel_median = list(map(lambda x: x / apte_median, apte_k_list))
        apte_k_list_rel_sum = list(map(lambda x: x / apte_sum, apte_k_list))
        apte_k_list_rel_max = list(map(lambda x: x / apte_max, apte_k_list))
        apre_k_list_rel_mean = list(map(lambda x: x / apre_mean, apre_k_list))
        apre_k_list_rel_median = list(map(lambda x: x / apre_median, apre_k_list))

        apte_rel_mean_mean = statistics.mean(apte_k_list_rel_mean)
        apte_rel_median_mean = statistics.mean(apte_k_list_rel_median)
        apte_rel_sum_mean = statistics.mean(apte_k_list_rel_sum)
        apte_rel_max_mean = statistics.mean(apte_k_list_rel_max)
        apre_rel_mean_mean = statistics.mean(apre_k_list_rel_mean)
        apre_rel_median_mean = statistics.mean(apre_k_list_rel_median)

        print("apte_sum: " + str(apte_sum))
        print("apte_mean: " + str(apte_mean))
        print("apte_median: " + str(apte_median))
        print("apre_sum: " + str(apre_sum))
        print("apre_mean: " + str(apre_mean))
        print("apre_median: " + str(apre_median))
        print("apte_rel_mean_mean: " + str(apte_rel_mean_mean))
        print("apte_rel_median_mean: " + str(apte_rel_median_mean))
        print("apte_rel_sum_mean: " + str(apte_rel_sum_mean))
        print("apte_rel_max_mean: " + str(apte_rel_max_mean))
        print("apre_rel_mean_mean: " + str(apre_rel_mean_mean))
        print("apre_rel_median_mean: " + str(apre_rel_median_mean))

        apte_k_list_rel_traj = (torch.tensor(apte_k_list) / torch.tensor(trajectory_length_list)).tolist()
        apre_k_list_rel_traj = (torch.tensor(apre_k_list) / torch.tensor(trajectory_length_list)).tolist()
        if apte_accumulative:
            apte_k_2_list_rel_traj = (torch.tensor(apte_k_2_list) / torch.tensor(trajectory_length_list)).tolist()

        apte_rel_traj_mean = statistics.mean(apte_k_list_rel_traj)
        apre_rel_traj_mean = statistics.mean(apre_k_list_rel_traj)
        if apte_accumulative:
            apte_k_2_list_rel_traj_mean = statistics.mean(apte_k_2_list_rel_traj)

        print("apte_rel_traj_mean: " + str(apte_rel_traj_mean))
        print("apre_rel_traj_mean: " + str(apre_rel_traj_mean))
        if apte_accumulative:
            print("apte_2_rel_traj_mean: " + str(statistics.mean(apte_k_2_list_rel_traj)))

        if save_metrics:
            f = open(os.path.join(save_folder, "metrics.txt"), 'w')  # overwrite the file
            f.writelines("apte_sum: " + str(apte_sum) + '\n')
            f.writelines("apte_mean: " + str(apte_mean) + '\n')
            f.writelines("apte_median: " + str(apte_median) + '\n')
            f.writelines("apre_sum: " + str(apre_sum) + '\n')
            f.writelines("apre_mean: " + str(apre_mean) + '\n')
            f.writelines("apre_median: " + str(apre_median) + '\n')
            f.writelines("apte_rel_traj_mean: " + str(apte_rel_traj_mean) + '\n')
            f.writelines("apre_rel_traj_mean: " + str(apre_rel_traj_mean) + '\n')
            f.writelines("apte_rel_mean_mean: " + str(apte_rel_mean_mean) + '\n')
            f.writelines("apte_rel_median_mean: " + str(apte_rel_median_mean) + '\n')
            f.writelines("apte_rel_sum_mean: " + str(apte_rel_sum_mean) + '\n')
            f.writelines("apte_rel_max_mean: " + str(apte_rel_max_mean) + '\n')
            f.writelines("apre_rel_mean_mean: " + str(apre_rel_mean_mean) + '\n')
            f.writelines("apre_rel_median_mean: " + str(apre_rel_median_mean) + '\n')
            metrics = str(apte_sum) + " "+ str(apte_mean) + " " + str(apte_median) + " " + str(apre_sum) + " "+ str(apre_mean) + " "+ str(apre_median) + " "+ str(apte_rel_traj_mean) + " "+ str(apre_rel_traj_mean) + " " + str(apte_rel_mean_mean) + " " + str(apte_rel_median_mean) + " "+ str(apte_rel_sum_mean) + " "+ str(apte_rel_max_mean) + " "+ str(apre_rel_mean_mean) + " "+ str(apre_rel_median_mean)
            if apte_accumulative:
                f.writelines("apte_2_rel_traj_mean: " + str(statistics.mean(apte_k_2_list_rel_traj)))
                metrics += " " + str(statistics.mean(apte_k_2_list_rel_traj))

            f.close()

        plt.plot(apte_k_list, label=method_label)

    if show_plots:
        plt.show()
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(save_image_folder, 'apte.pdf'))
    plt.close()

        # plt.figure()
        # plt.plot(apte_k_list, label="apte")
        # plt.xlabel("Loop length in #frames")
        # plt.ylabel("APTE")
        # plt.legend()
        # plt.savefig(os.path.join(save_folder, 'apte.pdf'))
        # if show_plots:
        #     plt.show()
        # plt.close()
        #
        # plt.figure()
        # plt.plot(apre_k_list, label="apre")
        # plt.xlabel("Loop length in #frames")
        # plt.ylabel("APRE [º]")
        # plt.legend()
        # plt.savefig(os.path.join(save_folder, 'apre.pdf'))
        # if show_plots:
        #     plt.show()
        # plt.close()
        #
        # plt.figure()
        # plt.plot(apte_k_list_rel_mean, label="APTE_rel_mean")
        # plt.plot(apte_k_list_rel_median, label="APTE_rel_median")
        # plt.plot(apre_k_list_rel_mean, label="APRE_rel_mean")
        # plt.plot(apre_k_list_rel_median, label="APRE_rel_median")
        # plt.xlabel("Loop length in #frames")
        # plt.legend()
        # plt.savefig(os.path.join(save_folder, 'APTE_rel.pdf'))
        # if show_plots:
        #     plt.show()
        # plt.close()
        #
        # plt.figure()
        # plt.plot(apte_k_list_rel_sum, label="apte_rel_sum")
        # plt.xlabel("Loop length in #frames")
        # plt.legend()
        # plt.savefig(os.path.join(save_folder, 'apte_rel_sum.pdf'))
        # if show_plots:
        #     plt.show()
        # plt.close()
        #
        # plt.figure()
        # plt.plot(apte_k_list_rel_max, label="apte_rel_max")
        # plt.xlabel("Loop length in #frames")
        # plt.legend()
        # plt.savefig(os.path.join(save_folder, 'apte_rel_max.pdf'))
        # if show_plots:
        #     plt.show()
        # plt.close()
        #
        # plt.figure()
        # plt.plot(apte_k_list_rel_traj, label="apte_rel_traj")
        # plt.plot(apre_k_list_rel_traj, label="apre_rel_traj")
        # plt.ylabel("Traslation drift in loop k per unit travelled")
        # plt.xlabel("Loop length in #frames")
        # plt.legend()
        # plt.savefig(os.path.join(save_folder, 'apte_rel_traj.pdf'))
        # if show_plots:
        #     plt.show()
        # plt.close()
        #
        # if apte_accumulative:
        #     plt.figure()
        #     plt.plot(apte_k_2_list_rel_traj, label="apte_2_rel_traj")
        #     plt.ylabel("Accumulated traslation drift in loop k per unit travelled")
        #     plt.xlabel("Loop length in #frames")
        #     plt.legend()
        #     plt.savefig(os.path.join(save_folder, 'apte_2_rel_traj.pdf'))
        #     if show_plots:
        #         plt.show()
        #     plt.close()
