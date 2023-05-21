import bpy
import os
import argparse
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output_folder", type=str, required=True)
parser.add_argument("-s", "--scene", type=str, required=True)
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

save_color_folder = os.path.join(args.output_folder, "color")
save_depth_folder = os.path.join(args.output_folder, "depth")
Path(save_color_folder).mkdir(parents=True, exist_ok=True)
Path(save_depth_folder).mkdir(parents=True, exist_ok=True)

total_steps = {"00000": 100, "00001": 1523, "00002": 1240, "00003": 3308, "00004": 4545, "00005": 1655, "00006": 1515,
               "00007": 2341, "00008": 3033, "00009": 8098, "00010": 3632, "00011": 4168, "00012": 4168, "00013": 3296,
               "00014": 23863, "00015": 7984, "00016": 5389, "00018": 10989, "00019": 9739}

scene = bpy.context.scene
scene.render.resolution_x = 1024
scene.render.resolution_y = 1024

# Find the existing Compositor depth output node
for node in scene.node_tree.nodes:
    if node.type == 'OUTPUT_FILE':
        output_node = node

output_node.base_path = save_depth_folder

range = range(0, total_steps[args.scene])
for step in range:
    scene.frame_set(step)
    scene.render.filepath = os.path.join(save_color_folder, '%010d.png' % step)
    bpy.ops.render.render(write_still=True)
