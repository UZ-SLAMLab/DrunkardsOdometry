""" Script to create convert poses from 4x4 style to quaternions.

Input .txt
timestamp, r_11, r_12, r_13, t_x, r_21, r_22, r_23, t_y, r_31, r_32, r_33, t_z

Output .csv:
1st Column: Timestamps
2-4th Columns: t_x, t_y, t_z [m]
5-8th Columns: q_x, q_y, q_z, q_w

"""

import csv
import numpy as np
from pyquaternion import Quaternion  # Hamilton quaternion convention. qw, qx, qy, qz
from scipy.spatial.transform import Rotation as R  # Hamilton quaternion convention. qx, qy, qz, qw
from tqdm import tqdm

input_pose_path = "/media/david/DiscoDuroLinux/Datasets/Matterport/train/poses/00008/level3/poses_w_hzc.txt"
output_pose_path = "/media/david/DiscoDuroLinux/Datasets/Matterport/train/poses/00008/level3/pose.txt"

# Read input poses
with open(input_pose_path) as f:
    input_poses = f.readlines()

# header = ['#timestamp', 't_x [m]', 't_y [m]', 't_z [m]', 'q_x', 'q_y', 'q_z', 'q_w']
f = open(output_pose_path, 'w')
# writer = csv.writer(f)
# writer.writerow(header)

for i in tqdm(range(len(input_poses))):
# for i in range(0, 150):
#     timestamp = '{:010d}'.format(i)
    pose = input_poses[i].split()
    timestamp = '{:010d}'.format(int(pose[0]))
    t_x = float(pose[4])
    t_y = float(pose[8])
    t_z = float(pose[12])
    rotation_matrix_3x3 = np.matrix([[float(pose[1]), float(pose[2]), float(pose[3])],
                                     [float(pose[5]), float(pose[6]), float(pose[7])],
                                     [float(pose[9]), float(pose[10]), float(pose[11])]])

    # print("Rotation matrix")
    # print(rotation_matrix_3x3)
    # print("Rotation matrix transpose")
    # print(rotation_matrix_3x3.transpose())
    # print("Rotation * transpose matrix")
    # print(np.matmul(rotation_matrix_3x3, rotation_matrix_3x3.transpose()))
    # print("Rotation matrix inverse")
    # print(np.linalg.inv(rotation_matrix_3x3))

    # Proyect 3x3 matrix to the special orthogonal group and vice versa to ensure rotation_matrix_3x3 is orthogonal. Following notation of https://arxiv.org/pdf/2004.00732.pdf
    U, SIGMA, V_t = np.linalg.svd(rotation_matrix_3x3)
    W = np.matrix(np.identity(3))
    if (np.linalg.det(np.matmul(U, V_t)) < 0):
        W[2, 2] = -1
    rotation_matrix_3x3 = np.matmul(np.matmul(U, W), V_t)
    # print("Rotation matrix proyected and unproyected to ensure orthogonality")
    # print(rotation_matrix_3x3)

    rotation_quaternions = Quaternion(matrix=rotation_matrix_3x3)
    q_w = rotation_quaternions[0]
    q_x = rotation_quaternions[1]
    q_y = rotation_quaternions[2]
    q_z = rotation_quaternions[3]

    # print("3x3 to quat by pyquaternion:")
    # print(rotation_quaternions)
    # rotation = R.from_matrix(rotation_matrix_3x3)
    # print("3x3 to quat by scipy:")
    # print(rotation.as_quat())

    # row = [timestamp, t_x, t_y, t_z, q_x, q_y, q_z, q_w]
    # writer.writerow(row)
    row = timestamp + ' ' + str(t_x) + ' ' + str(t_y) + ' ' + str(t_z) + ' ' + str(q_x) + ' ' + str(q_y) + ' ' + str(q_z) + ' ' + str(q_w)
    f.writelines(row + '\n')
f.close()





# # From rotation matrix to quaternion
# pose = "0.43833011388778687 0.017149293795228004 -0.8986504077911377 0.02616182714700699 -0.898807942867279 0.004666342865675688 -0.43831783533096313 -0.0014262627810239792 -0.0033234308939427137 0.9998420476913452 0.017459316179156303 -1.200169563293457".split()
# t_x = float(pose[3])
# t_y = float(pose[7])
# t_z = float(pose[11])
# rotation_matrix_3x3 = np.matrix([[float(pose[0]), float(pose[1]), float(pose[2])],
#                                  [float(pose[4]), float(pose[5]), float(pose[6])],
#                                  [float(pose[8]), float(pose[9]), float(pose[10])]])
# print('Rotation matrix')
# print(rotation_matrix_3x3)
#
# # U, SIGMA, V_t = scipy.linalg.svd(rotation_matrix_3x3)
# U, SIGMA, V_t = np.linalg.svd(rotation_matrix_3x3)
# W = np.matrix(np.identity(3))
# if (np.linalg.det(np.matmul(U, V_t)) < 0):
#     W[2, 2] = -1
# rotation_matrix_3x3 = np.matmul(np.matmul(U, W,), V_t)
# print('Rotation matrix proyected and unproyected')
# print(rotation_matrix_3x3)
#
# rotation_quaternions = Quaternion(matrix=rotation_matrix_3x3)
# q_w = rotation_quaternions[0]
# q_x = rotation_quaternions[1]
# q_y = rotation_quaternions[2]
# q_z = rotation_quaternions[3]
# row = ['timestamp', t_x, t_y, t_z, q_x, q_y, q_z, q_w]
# print('Rotation in quaternion')
# header = ['#timestamp', 't_x [m]', 't_y [m]', 't_z [m]', 'q_x', 'q_y', 'q_z', 'q_w']
# print(header)
# print(row)
#
# # From quaternion to rotation matrix
# rotation = R.from_quat([q_x, q_y, q_z, q_w])
# print('Rotation as matrix')
# print(rotation.as_matrix())
# print('Rotation as quaternion')
# print(rotation.as_quat())
