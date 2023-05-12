import bpy
import os
from mathutils import *
import numpy as np

""" Use this code inside a blender project to record camera poses."""

prefix_pose = "/media/david/ExtDisk1/drunkSLAM/poses_level4/00000/"
# prefix_image = "/media/david/ExtDisk1/drunkSLAM/poses_level4/00000/color/"
step_count = 3816
meshName = "chunk000_group000_sub007"


# def save_camera_pose_initial(camera_pose):
#    global inverted_camera_pose_initial
#    inverted_camera_pose_initial = camera_pose.copy()

def get_camera_pose(cameraName, objectName, scene, frameNumber):
    if not os.path.exists(prefix_pose):
        os.makedirs(prefix_pose)

    # OpenGL to Computer vision camera frame convention
    M = Matrix().to_4x4()
    M[1][1] = -1
    M[2][2] = -1

    cam = bpy.data.objects[cameraName]
    # object_pose = bpy.data.objects[objectName].matrix_world

    # Normalize orientation with respect to the scale
    # object_pose_normalized = object_pose.copy()
    # object_orientation_normalized = object_pose_normalized.to_3x3().normalized()
    # for i in range(3):
    #    for j in range(3):
    #        object_pose_normalized[i][j] = object_orientation_normalized[i][j]

    if frameNumber == 0:
        camera_pose_world = Matrix().to_4x4()
        # save_camera_pose_initial(cam.matrix_world.inverted())

    camera_pose_cv = (M @ cam.matrix_world.inverted()).inverted()
    # camera_pose_world = cam.matrix_world
    # camera_pose_cv_tutorial = M @ cam.matrix_world.inverted() @ object_pose_normalized
    # print("camera_pose_w_oglc_tutorial:\n", camera_pose_cv_tutorial)
    print("camera_pose_w_hzc:\n", camera_pose_cv)
    # print("camera_pose_w_oglc:\n", camera_pose_world)

    timestamp = '{:010d}'.format(frameNumber)

    filename_cv = prefix_pose + "poses_w_hzc.txt"
    with open(filename_cv, 'a') as f:
        f.write(timestamp + " " + \
                str(camera_pose_cv[0][0]) + " " + str(camera_pose_cv[0][1]) + " " + str(
            camera_pose_cv[0][2]) + " " + str(camera_pose_cv[0][3]) + " " + \
                str(camera_pose_cv[1][0]) + " " + str(camera_pose_cv[1][1]) + " " + str(
            camera_pose_cv[1][2]) + " " + str(camera_pose_cv[1][3]) + " " + \
                str(camera_pose_cv[2][0]) + " " + str(camera_pose_cv[2][1]) + " " + str(
            camera_pose_cv[2][2]) + " " + str(camera_pose_cv[2][3]) + "\n")

    # filename_world = prefix_pose + "poses_w_oglc.txt"
    # with open(filename_world, 'a') as f:
    #    f.write(timestamp + " " + \
    #    str(camera_pose_world[0][0]) + " " + str(camera_pose_world[0][1]) + " " + str(camera_pose_world[0][2]) + " " + str(camera_pose_world[0][3]) + " " + \
    #            str(camera_pose_world[1][0]) + " " + str(camera_pose_world[1][1]) + " " + str(camera_pose_world[1][2]) + " " + str(camera_pose_world[1][3]) + " " + \
    #            str(camera_pose_world[2][0]) + " " + str(camera_pose_world[2][1]) + " " + str(camera_pose_world[2][2]) + " " + str(camera_pose_world[2][3])+ "\n")

    # filename_cv_tutorial = prefix_pose + "poses_w_hzc_tutorial.txt"
    # with open(filename_cv_tutorial, 'a') as f:
    #    f.write(str(camera_pose_cv_tutorial[0][0]) + " " + str(camera_pose_cv_tutorial[0][1]) + " " + str(camera_pose_cv_tutorial[0][2]) + " " + str(camera_pose_cv[0][3]) + " " + \
    #            str(camera_pose_cv_tutorial[1][0]) + " " + str(camera_pose_cv_tutorial[1][1]) + " " + str(camera_pose_cv_tutorial[1][2]) + " " + str(camera_pose_cv[1][3]) + " " + \
    #            str(camera_pose_cv_tutorial[2][0]) + " " + str(camera_pose_cv_tutorial[2][1]) + " " + str(camera_pose_cv_tutorial[2][2]) + " " + str(camera_pose_cv[2][3])+ "\n")

    return


def my_handler(scene):
    frameNumber = scene.frame_current
    print("\n\nFrame Change", scene.frame_current)
    get_camera_pose("Camera", meshName, scene, frameNumber)


scene = bpy.context.scene
for step in range(0, step_count):
    # Set render frame
    scene.frame_set(step)

    # Set filename and render
    # if not os.path.exists(prefix_image):
    #    os.makedirs(prefix_image)
    # scene.render.filepath = (prefix_image + '%010d.png') % step
    # bpy.ops.render.render(write_still=True)

    my_handler(scene)