import shutil
from glob import glob
import os
from tqdm import tqdm

"""Script to clone backward the Hamlyn Dataset test scenes 1 and 17."""

root = "/.../hamlyn_dataset"

file_extension_dict = {"color": "jpg", "depth": "png"}

for scene in ["test1", "test17"]:
    for data in ["color", "depth"]:
        input_image_folder = "/.../{}/{}".format(scene, data)
        output_image_folder = "/.../{}_backward/{}".format(scene, data)
        file_extension = file_extension_dict[data]

        input_images = sorted(glob(os.path.join(input_image_folder, "*.{}".format(file_extension))))
        first_frame_number = int(os.path.splitext(os.path.basename(input_images[0]))[0])
        last_frame_number = int(os.path.splitext(os.path.basename(input_images[-1]))[0])

        for i, input_image in tqdm(enumerate(input_images)):
            output_image = os.path.join(output_image_folder, os.path.join('{:010d}.{}'.format(last_frame_number - i, file_extension)))
            shutil.copy(input_image, output_image)
