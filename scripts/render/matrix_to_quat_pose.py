import os
import numpy as np
from pyquaternion import Quaternion  # Hamilton quaternion convention: qw, qx, qy, qz
import argparse
from tqdm import tqdm

""" Script to convert poses from 3x4 matrix style to quaternions.

Input .txt
timestamp, r_11, r_12, r_13, t_x, r_21, r_22, r_23, t_y, r_31, r_32, r_33, t_z

Output .txt:
1st Column: Timestamps
2-4th Columns: t_x, t_y, t_z 
5-8th Columns: q_x, q_y, q_z, q_w
"""

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_pose_path", type=str, required=True)
parser.add_argument("-o", "--output_pose_path", type=str, required=True)
args = parser.parse_args()
print(args)

input_pose_path = os.path.join(args.input_pose_path, "pose_matrix.txt")
output_pose_path = os.path.join(args.output_pose_path, "pose.txt")

print("Converting matrix to quaternion poses")

# Read input poses
with open(input_pose_path) as f:
    input_poses = f.readlines()

f = open(output_pose_path, 'w')

for i in tqdm(range(len(input_poses))):
    pose = input_poses[i].split()
    timestamp = '{:010d}'.format(int(pose[0]))
    t_x = float(pose[4])
    t_y = float(pose[8])
    t_z = float(pose[12])
    rotation_matrix_3x3 = np.matrix([[float(pose[1]), float(pose[2]), float(pose[3])],
                                     [float(pose[5]), float(pose[6]), float(pose[7])],
                                     [float(pose[9]), float(pose[10]), float(pose[11])]])

    # Proyect 3x3 matrix to the special orthogonal group and vice versa to ensure rotation_matrix_3x3 is orthogonal.
    # Following notation of https://arxiv.org/pdf/2004.00732.pdf
    U, SIGMA, V_t = np.linalg.svd(rotation_matrix_3x3)
    W = np.matrix(np.identity(3))
    if (np.linalg.det(np.matmul(U, V_t)) < 0):
        W[2, 2] = -1
    rotation_matrix_3x3 = np.matmul(np.matmul(U, W), V_t)

    rotation_quaternions = Quaternion(matrix=rotation_matrix_3x3)
    q_w = rotation_quaternions[0]
    q_x = rotation_quaternions[1]
    q_y = rotation_quaternions[2]
    q_z = rotation_quaternions[3]

    row = timestamp + ' ' + str(t_x) + ' ' + str(t_y) + ' ' + str(t_z) + ' ' + str(q_x) + ' ' + str(q_y) + ' ' + str(q_z) + ' ' + str(q_w)
    f.writelines(row + '\n')
f.close()

print("\n     --> Done!")