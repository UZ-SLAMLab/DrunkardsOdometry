import shutil
from glob import glob
import os
from tqdm import tqdm

input_image_folder = "/home/david/datasets/hamlyn_for_drunk_paper/test22/color"
output_image_folder = "/home/david/datasets/hamlyn_for_drunk_paper/test22_backward/color"
file_extension = "jpg"

input_images = sorted(glob(os.path.join(input_image_folder, "*.{}".format(file_extension))))
first_frame_number = int(os.path.splitext(os.path.basename(input_images[0]))[0])
last_frame_number = int(os.path.splitext(os.path.basename(input_images[-1]))[0])

for i, input_image in tqdm(enumerate(input_images)):
    output_image = os.path.join(output_image_folder, os.path.join('{:010d}.{}'.format(last_frame_number - i, file_extension)))
    shutil.copy(input_image, output_image)
