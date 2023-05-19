import bpy
import os
from mathutils import *
import argparse
import sys


def get_camera_pose(cameraName, frameNumber):
    if not os.path.exists(prefix_pose):
        os.makedirs(prefix_pose)

    # OpenGL to Computer vision camera frame convention
    M = Matrix().to_4x4()
    M[1][1] = -1
    M[2][2] = -1

    cam = bpy.data.objects[cameraName]

    camera_pose_cv = (M @ cam.matrix_world.inverted()).inverted()

    # Pose world-to-hzcamera (openCV) r11, r12, r13, tx, r21, r22, r23, ty, r31, r32, r33, tz
    filename_cv = os.path.join(prefix_pose, "pose_matrix.txt")
    timestamp = '{:010d}'.format(frameNumber)
    with open(filename_cv, 'a') as f:
        f.write(timestamp + " " + \
                str(camera_pose_cv[0][0]) + " " + str(camera_pose_cv[0][1]) + " " + str(
            camera_pose_cv[0][2]) + " " + str(camera_pose_cv[0][3]) + " " + \
                str(camera_pose_cv[1][0]) + " " + str(camera_pose_cv[1][1]) + " " + str(
            camera_pose_cv[1][2]) + " " + str(camera_pose_cv[1][3]) + " " + \
                str(camera_pose_cv[2][0]) + " " + str(camera_pose_cv[2][1]) + " " + str(
            camera_pose_cv[2][2]) + " " + str(camera_pose_cv[2][3]) + "\n")


def my_handler(scene):
    frameNumber = scene.frame_current
    get_camera_pose("Camera", frameNumber)


def progressbar(current_value,total_value,bar_lengh,progress_char):
    """https://stackoverflow.com/a/75033230"""
    percentage = int((current_value/total_value)*100)
    progress = int((bar_lengh * current_value ) / total_value)
    loadbar = "Progress: [{:{len}}]{}%".format(progress*progress_char, percentage, len=bar_lengh)
    print(loadbar, end='\r')


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pose_folder", type=str, required=True)
parser.add_argument("-s", "--scene", type=str, required=True)
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

prefix_pose = args.pose_folder
scene_id = args.scene

total_steps = {"00000": 100,"00001": 1523,"00002": 1240,"00003": 3308,"00004": 4545,"00005": 1655,"00006": 1515,
               "00007": 2341,"00008": 3033,"00009": 8098,"00010": 3632,"00011": 4168,"00012": 4168,"00013": 3296,
               "00014": 23863,"00015": 7984,"00016": 5389,"00018": 10989,"00019": 9739}

scene = bpy.context.scene
print("\nCamera transformation world-to-hzc from current frame to the following")

range = range(0, total_steps[scene_id])

for step in range:
    # Set render frame
    scene.frame_set(step)
    my_handler(scene)

    progressbar(step+1, len(range), 30, '■')
