import bpy
import os
from mathutils import *
import numpy as np
import argparse
import sys
from pyquaternion import Quaternion  # Hamilton quaternion convention. qw, qx, qy, qz


def get_camera_pose(cameraName, frameNumber):
    if not os.path.exists(prefix_pose):
        os.makedirs(prefix_pose)

    # OpenGL to Computer vision camera frame convention
    M = Matrix().to_4x4()
    M[1][1] = -1
    M[2][2] = -1

    cam = bpy.data.objects[cameraName]

    pose = (M @ cam.matrix_world.inverted()).inverted()
    print("camera_pose_w_hzc:\n", pose)

    t_x = float(pose[0][3])
    t_y = float(pose[1][3])
    t_z = float(pose[2][3])
    rotation_matrix_3x3 = np.matrix([[float(pose[0][0]), float(pose[0][1]), float(pose[0][2])],
                                     [float(pose[1][0]), float(pose[1][1]), float(pose[1][2])],
                                     [float(pose[2][0]), float(pose[2][1]), float(pose[2][2])]])

    # Ensure invertibility
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

    timestamp = '{:010d}'.format(frameNumber)

    filename_cv = prefix_pose + "pose.txt"  # Pose world-to-hzcamera (openCV) quaternions tx, ty, tz, qx, qy, qz, qw
    with open(filename_cv, 'a') as f:
        f.write(timestamp + " " + \
                str(t_x) + ' ' + str(t_y) + ' ' + str(t_z) + ' ' +
                str(q_x) + ' ' + str(q_y) + ' ' + str(q_z) + ' ' + str(q_w) + "\n")

    return


def my_handler(scene):
    frameNumber = scene.frame_current
    print("\n\nFrame Change", scene.frame_current)
    get_camera_pose("Camera", frameNumber)


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pose_folder", type=str, required=True)
parser.add_argument("-n", "--num_images", type=int, required=True)
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

prefix_pose = args.pose_folder
step_count = args.num_images


scene = bpy.context.scene
for step in range(0, step_count):
    # Set render frame
    scene.frame_set(step)

    my_handler(scene)
