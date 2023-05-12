import bpy
import os
from mathutils import *
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pose_folder", type=str, required=True)
parser.add_argument("-c", "--color_folder", type=str, required=True)
parser.add_argument("-n", "--num_images", type=int, required=True)
parser.add_argument("-m", "--mesh_name", type=str, required=True)
args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])

prefix_pose = args.pose_folder
prefix_image = args.color_folder
step_count = args.num_images
meshName = args.mesh_name


def get_camera_pose(cameraName, objectName, scene, frameNumber):
    if not os.path.exists(prefix_pose):
        os.makedirs(prefix_pose)     
        
    # OpenGL to Computer vision camera frame convention
    M = Matrix().to_4x4()
    M[1][1] = -1
    M[2][2] = -1
    
    cam = bpy.data.objects[cameraName]
            
    if frameNumber == 0:
        camera_pose_world = Matrix().to_4x4()         
          
    camera_pose_cv = (M @ cam.matrix_world.inverted()).inverted()         
    print("camera_pose_w_hzc:\n", camera_pose_cv)    

    timestamp = '{:010d}'.format(frameNumber)  

    filename_cv = prefix_pose + "poses_w_hzc.txt"    
    with open(filename_cv, 'a') as f:
        f.write(timestamp + " " + \
                str(camera_pose_cv[0][0]) + " " + str(camera_pose_cv[0][1]) + " " + str(camera_pose_cv[0][2]) + " " + str(camera_pose_cv[0][3]) + " " + \
                str(camera_pose_cv[1][0]) + " " + str(camera_pose_cv[1][1]) + " " + str(camera_pose_cv[1][2]) + " " + str(camera_pose_cv[1][3]) + " " + \
                str(camera_pose_cv[2][0]) + " " + str(camera_pose_cv[2][1]) + " " + str(camera_pose_cv[2][2]) + " " + str(camera_pose_cv[2][3])+ "\n")
                               
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
    if not os.path.exists(prefix_image):
        os.makedirs(prefix_image)
    scene.render.filepath = (prefix_image + '%010d.png') % step
    bpy.ops.render.render(write_still=True)
    
    my_handler(scene)
