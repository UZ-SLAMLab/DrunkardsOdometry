import argparse
import gdown
import os
from glob import glob
import zipfile
from pathlib import Path

# root drive folder https://drive.google.com/drive/folders/1mLnAi4KTBc_8C0InIQgtRvKVQSrQSSDy?usp=sharing

def download(args):

    if args.all:
        args.scenes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19]
        if not(args.color or args.depth or args.optical_flow or args.normal or args.pose):
            args.color = args.depth = args.optical_flow = args.normal = args.pose = True
        if not args.levels:
            args.levels = [0, 1, 2, 3]
    else:
        assert args.scenes, "User must specify which scenes to download"
        assert args.levels, "User must specify which levels to download"
        assert args.color or args.depth or args.optical_flow or args.normal or args.pose, "User must download at least one of the followings: --color, --depth, --optical_flow, --normal, --pose"

    args.output_folder = os.path.join(args.output_folder, 'drunk_dataset')
    print("Downloading in " + args.output_folder)

    if 0 in args.scenes:
        scene = 0
        if 0 in args.levels:
            level = 0
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1suBizF8KJWhiMxigjVpKCs-q_bESSq__/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1FT1PwYsZ1BrJ5DP1ZpEIzGC3BlIzm0wh/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1OFTeIUMn-19fo6sVF69O-FK_PVzaUX2v/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))                
                url = "https://drive.google.com/file/d/1BIb-eJHSD-tB7YK0La0i0k8TpPdd24kQ/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1Xz31d6P9cntEc4itxiWzxmCfaLQng_FX/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 1 in args.levels:
            level = 1
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1AfOfIBnWVVeL0s_FVClDz23Ubc3FZyb-/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1a2O66acAirtEOJc3Uu4M9UOej6jQ5Lw1/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1RbbLJHl0DVif0RJTxVeDfLOk8xHV-VDY/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1vEbJ2QG2XJhh_zwXbl4LeX_OSHx9Kj4G/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1i1GYgWuZUWY-D6D6aBT9cC36aN4XQpzn/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 2 in args.levels:
            level = 2
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1-mynvEztGwUBicyOM-4-VH0JRmAZyQKg/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/12VDVGteZsB-DBMru5ln1it4BU69ekJLC/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1HT5DUYMRz-yRGnCXPNrzbVpKLX-XHBBy/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1zDTEAILhcGxz3AMJDh8PCJt6EBOhWBx7/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1gFSf6-hy1mG-3rT_sIRUVbv74imFPvWj/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 3 in args.levels:
            level = 3
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1RmsDDOaVb0EscM_Ez1mfZFO9WbS3Delm/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/14yFuLLr6sd3C19jrfvTjUzRbjvhobQhB/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1VGFmUdfIlVWEqkOGk-yyjcG7VoXVPhWh/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1nZbhtE43RApD1U_IdxRN5Rk0zq5tBN3T/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1oj4Sg4jM2mYXEjXP1LpqcZPheED_PC6V/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)

    if 1 in args.scenes:
        scene = 1
        if 0 in args.levels:
            level = 0
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1KkGa49v101l7mNv3YTseK0Uzn-QHCiNt/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1ax0D0c-GETCHLe-g-Y8QNGWrEzcQhu8D/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1nNgW1XMcNd2XNvrhi5I_Wyhfk8wrM56m/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1FL4x3mpdnPCQ6TrXSuFWIL8dRiMTZE3I/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1Xwx7Bt54NgQkIak0zMMyJG9L8OQOEY9W/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 1 in args.levels:
            level = 1
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1sEi7KwRd77l6OjtyoA4fr91GTIwFy78M/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1WGbtw-QPspBEHugBXu5oLjjmloGaRpvl/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1Fh9HiWSlfuYaGqzbw0fzklidFoPNDLdI/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1WIU_93GjrgJiCjmZ-wthy-bw9-tyGmTT/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1_QQepIoqw6l0OksbT68O-tUu-C9nWRLp/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 2 in args.levels:
            level = 2
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1mccplbeC7Jgtow85asbtTyKwbhlSGNVY/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1Vmmn-gbT9nsBMd1DOkbLyMBlr_AMK0cJ/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1IVLWaLl302Ty7yqe0adTEl9o-dRou71D/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1EJCgr7--dXbBNgIOCA6gr9YEA3ijwvgb/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1leuyRofMJFmnVsAbqjHNyl_V48c4IWtG/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 3 in args.levels:
            level = 3
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1yXt76dzMgocc3fvxIGhuX6WcB-g-3VCS/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1HnbbAFAxC180p5XCou4XI9wozocthqba/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/14x5MIupb7_vyBVK3dSqwcLtaOFXbTs3q/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/18v_sgrAr3ZVjIXE24fSy-_4IFAsHnR_o/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/10YHk1oheKV-6YmGYOBz8ylAOBIWQxe2y/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)

    if 2 in args.scenes:
        scene = 2
        if 0 in args.levels:
            level = 0
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1TpX_qRPF5gFSh13VJwCVgxLdwi4zyVo3/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1sxH45ROHl4_JreQvrI6wA_6_K6E14T6I/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1OIGDaC6UkKruhi8PkEuU4FRHK4B05P-7/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1jtAgxHQ41W9FaAQWabY4V1zCEBWl5k4B/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1vhKTtIdBTjKYRLsck97-i_LR6X8TJ7rW/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 1 in args.levels:
            level = 1
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1WG26Uij0iaOyJYjO81CIKBbJ_StPF1iN/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1pU6r8gmAx8bzsxdm68EiACW-7YLh0u5O/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1NVtBwOSEAwmDv3rP1wiGzf27H5m2b8QM/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1p01vgpAYs3VqHRAS47ywgNNEYq5ugJwE/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1swDij33GNEZWGBtijQ60EEnkgjPQ9ZzH/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 2 in args.levels:
            level = 2
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/14fscIUO10h9vlMHXBIlR7LooR2u_-X0q/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/15OOYBdu8yXCOCyx_V2L0yKBApt5lHwp7/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1cJyvgeUk3gS1UpflNZ5FJW07dAFxSPRs/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/12_tTypyXgFOU30yPdQf4s616E-B-o5Gj/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1SsLdt-eMrkj5zMUEg1UIFTZ9jXyVZKOu/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 3 in args.levels:
            level = 3
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1CXbKVnZlFZ6J1WbZS1ku_kIIS1gc4hcj/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1hFJSGFr5ufOcd2k0CqHqfjHzf3P3AQBB/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1lMJXjLJwq6YZsiOMV6KJ5uoL2pGRmMJG/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1t9wX1254MY5lSJSXHSmjicFxAA6OZ00S/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1VLHNUkw_TMpfYOrn_NzLgac42r4XoHdl/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)

    if 3 in args.scenes:
        scene = 3
        if 0 in args.levels:
            level = 0
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1LywHRlvglSzOcAI1K34IzRci7T_XAiqk/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1ElJb6ws6GALvA7C4jTdQoCGyq3BbDt0T/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1s17mwBeRkiL-_LDelLDhkx6GmUJKf3gq/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1zldQSfXv5gJ9Ql79e99RBB8Vql2IoFIC/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1GX0SrCCA50ZFGpVpeOWbvCGTqk6Lf_sk/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 1 in args.levels:
            level = 1
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1gjy6xNqZ06fI_d4UnsBSsfRfIYiN3GRZ/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1giVnJwSZpmQlhl_Okwp0-DVnVUS2jY52/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1F9-Vo33OLApEVdneNXCWBeXLteU-cOch/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1ZSip6KIjJRvBWHR5cp9uAy_6HXGJSYJl/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/14BBS06PGyeFzwp-qo_uYQfTGeiPFr_zZ/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 2 in args.levels:
            level = 2
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1IY6bIofFdntJQRNOQ5CFPMXYvs9UDyli/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1Uwb42GQMo_ooMF6B4IJWjdrHUQnHouEj/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1lc3ptf9A_Sk_5PB9XPw5_Zjy75nSpqbY/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1xc0oqO-UEnfQps5mBGODuh5wGWiih1_b/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1OGefDyaCpv3AWmIyToJiqcetoNxcUpM0/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 3 in args.levels:
            level = 3
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1FnwRWYmxkef2VGYT_DCzxqK1h1Ocflek/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1KOc2XVobgb76aQmTI9cYkJX0rB0vecpu/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1Q4zQJ56fCeA9P0WXfAfkv20bMjr80a-u/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1pV9Gvlbqg1BMAN3UHkS4SB7v26qcbaz_/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1K61ZMKTiXgdzQlIbPYhObJUvUm1wJg0p/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)

    if 4 in args.scenes:
        scene = 4
        if 0 in args.levels:
            level = 0
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1Dt_3bWxdE3Jwe6wVmJU13aveF6V35yq7/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1OhkOnguYTRrvQWsQkyQS2Nlx_vZuztV7/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/18EvQEBwqOx6DXBf1-P-bTfFg7Tgdl26u/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1ST16ZjA9hPeukjecSLyhTlCE8X1HxsVu/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1bhf_H2ofKtMuMbkADXnDl-MkLLr4vRxh/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 1 in args.levels:
            level = 1
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1HYLRZX5Fpj0OneDASnUrP6S0dbPHRJW1/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1R1zScsEKX7U9jgXIO-bwJTHTVxCwgLCy/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1PCbVrovqqYJUBNdgox1gK-PyM-MlUpMe/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1cJj2Y4tGCjhkO4rsXOBs9SfCQCd7c-Yx/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1pEoxCVR1uEvaQKzfDD09jIyWdSDywW3b/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 2 in args.levels:
            level = 2
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1Zh6RUFvN43MyIwOIvPz1mey7m4a4IRZg/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1koQPMoHwN8GhfW4pE0uaA3Yio1orZdfe/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1oB0F3iMNQKyAp4bXQ9CkzGZNV9PcVOmb/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1rE4jYVNHcByY0GtiJc5tYZElaF0ayRL2/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1FEO2lXJ2S8tzYu0rSNPp1UqQcX1JpIPN/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 3 in args.levels:
            level = 3
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/18DhwN6zrhhHMjtqgjsTPBkSZCqHcO1WU/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1oBmB7ojfAt_iNk2SkoOwbJhB4CfrhPnm/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1idF-fn08TecFAnxZD3H_fNnXOMO7J9FV/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/14MoGySJA5YCyCfGtQIBpIKXzfrGNBqn7/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1BHK331OnaauPf7FnJ6pvoCeHRXc4XPRk/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)

    if 5 in args.scenes:
        scene = 5
        if 0 in args.levels:
            level = 0
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1Tl_5N5ZYV4qWfhnvTn4VlXrk2lFhz1yb/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/17nI44TS_DDvzWsVAhEZ11A_N_GntVIae/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1GjRGS1WD7-OyuFbt8-Osmtr0xFZ20tyG/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1SZGxPR_HhmEq143EgnVWEvUMZCgTsAU-/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1C8M38AWZcfmhDhGg-vwmNbsGmRy44Slf/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 1 in args.levels:
            level = 1
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1yepWEPFroEq90pjw5zO08NIyxxco0_CY/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1QVAm4v8mf2DLJAvy6XiHk18iUespb1D3/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1E2HBEbaZnJJc5W5VmuYm5VbS_epnjjrI/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1xsc0Yjt00fPrc6tJloAcTFx_WxWiltWH/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1CyyhT3g1f5IBcNQypEVQSllzTdYUJl5r/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 2 in args.levels:
            level = 2
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1bYZ79LUZO0IA_mK8C6Dr_GMe1HwUoYJr/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1XMIuR3v6YYLpjijeMVPHl-fwLSY7Fie8/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1NZMIweYY7D2CjbJbRTZIoq0Lglzq4BKT/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1Grnw-czz0aM4JMJp8uf7Ub-DjFW9mFfL/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1w7Tir3EJBPJp4wXueg1xeJLE9wMlcRTS/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 3 in args.levels:
            level = 3
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1AvQk3gRxflYzLm7HgcMfqxe554Y2W_CD/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1aqeKWkBn8zjJ8cSAMT8j1PiWhgoBrFNF/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1kyvZ7z2AF2ewa7BUcOGQVTn2iIc0irku/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1qrWir_lnymbgWA8-5mmXjvbj-xkGTgcZ/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/127M0PnvZyjzJ72lWNEgjXw0nZcPUTaEk/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)

    if 6 in args.scenes:
        scene = 6
        if 0 in args.levels:
            level = 0
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1dl4as2FesGZLYJnANNlxyZR3RMOOsV24/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1G35lDheLHZ9iwNU6VR7ySSBVAxrdptAn/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1XjrSvYub1LEKc9RmC6mRknLhF5Udf5nJ/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1G4rLLO7Lst2Gcwi2GloiU-Cim4EbJ_rJ/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1wDwt1U02ohxragZDApf-jiflMfSFVuiU/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 1 in args.levels:
            level = 1
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1JkmATYym0sYKalcSXudIuBGfpeNHPuON/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1kQcKbFgNnr7doERJuT7E0chcYRh_oUGE/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1d8dLoV7Pnh8ZArBFE7JDAuoHe_47rDYL/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1tzYKfmeFnnlKYbzo8PeVkVUnRTeoYwbU/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1MB9a1SMrR4_ccjfXbpGap5kbx4u9qiUq/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 2 in args.levels:
            level = 2
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1gEksSZ1qodVG16pxT0dDMEnn-cuS9dzD/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1WxNuGwsyCf3mdXhlczL_KDNHvYnrFxdO/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1NQ9R7V2-Ichm5em2trWX-4QpgRzbIPz8/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/15Wrl7X9O7YXXBmWvmd2AwNKPEYLszt39/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1f84RMcN8Am64-pZTbCYeI-ehitUkYQoJ/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 3 in args.levels:
            level = 3
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1tXZvzZFU6Gl7qEyb_QnoZGQCmrHnpsOJ/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1xEkh9LwI4v42kqY7E5fw04OfuA2w6W0S/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1CmHLwVLQmvxtZSHR1I3IbmHGlycb8APz/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1IGZ60DG18IXamv2Sf1oxI5WI6piG8681/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/172-yrz2QsVVCNrFKQisR_SWNob5KfWal/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)

    if 7 in args.scenes:
        scene = 7
        if 0 in args.levels:
            level = 0
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1oW1BKyEpTwWOgOy-AFqeNJRMNIqLKsdY/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/15q092vAkuHUrv7siGkezHKpuy9UvRrP1/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1RU9BkO_SlqmzA-jXMHkhd3f52tPhAkjh/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1LTF1uXsGzanL_VAQSywfRf42ScPQdNUu/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1NlVrsPaSB6d1CAtG9c0WKkr3sF0sJyWu/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 1 in args.levels:
            level = 1
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1yCZzs5Go3lf4GZd9B0-axT0RhjQDmMKQ/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1QbVG1Eiih6R0QJIevj5oQ2d0Yf6sCOOR/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1Vxrw_14K-ZkjzKQHF_7_UzEEP8_7NNOa/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/12gJAcBiuVszl4xgSgHuTXtq7rBOgZQIs/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1kxfWwKeA-hpPQ7ylgdlKpBIzPG7npX8D/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 2 in args.levels:
            level = 2
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1bCmf1e3qBU0PPmOFntNq5LBJjuS2YLjZ/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/10EVvBooRJWU1doYYaYQ5WYww2O-LIXlZ/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1fK0nYTbJM9Yb8ea4PWPO7G6NZ6UmJyLC/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/15ZDr-La-s9heFMWQlUk3aRl5r6hK0pqv/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/19ATR5DoV0zJv-WNR3FAd-AAxnC2M6dI2/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 3 in args.levels:
            level = 3
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1lhpSglDKyh_weDtgxj5DgQeMh4YtVaTU/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1ZYQD40lIxLzpgxn-io2He0m3ASfOqz-t/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1yNLlKQSl3hs5-I7lD2p2f4lDMaUTcMKa/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1CAHyGKrx7aqzvQ5_ZLcpXpH3vkSSOZmp/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1LCwaUsGBHlkykSOvztYV-3STw3NGLHRq/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)

    if 8 in args.scenes:
        scene = 8
        if 0 in args.levels:
            level = 0
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/17ZnDgpHiOcqH5_BnRnIDWByOcedIJZt8/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1MQ9aSgSjjVJh4CvIJGJqMBjjP5Nck4Cw/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1ErBYlLvI37aXzoZK6kPQWEkfEnWEjEdG/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1lAYlSMjnNMrt24yvabiC8lTpadYp9NDE/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1rfb9RIh4E6ta0MFz_9XTHdBKC-1KzwET/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 1 in args.levels:
            level = 1
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1Y5nBWF3XDrTdo--khCCv4MGbaZlyX82X/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1zyVZlAYmhLCNXknvJPYWPctBymVCri1Q/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/16f10jv3to_x8byXCiX5dB7U0biPqF0u5/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/12AVxhlWJZtpeTuDwFWhD2tbCISmIHeWy/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1aPyYyMqD14GPqtAQqZ7QbLFv5qn0pG3O/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 2 in args.levels:
            level = 2
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/17yV1jZieoEUdb8P3oRD66J273456ZDG0/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1fVA1j2ZyAj5zJTX9gQMzLwNcaIpomEMl/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/17o5CqeZ9d8lsTESiU4klMcgdiEeeVvzT/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1CbIQF2zv9CfXA48HRPUafpS3c-lqUERm/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/17kAEE0RdkGKUaE0Y7hQ84OTEBO2qKW_R/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 3 in args.levels:
            level = 3
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1YfBozVtO2di7NSVwHpQwxTR3U1oB5C4V/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1PWdWl-m1wdg8NxEH4KN3cgGDZp5Ecwdo/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1VpjqxKxM21myXWgr7cuQOd-zely9_KnC/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1TLRcM4JUP3Zvy5tE_n8VKT6bENat_fUQ/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/17adWE9oTUvJnJgtFsSSHn9CdBmeolwAj/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)

    if 9 in args.scenes:
        scene = 9
        if 0 in args.levels:
            level = 0
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1TZ4zaFMq73CymzGDymG9HNBbDSZXgwkk/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/15Gg6dXiSSRZ8vDpYr-FIvtJyUkOzLgKw/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1-yO50TvvajXVtUsEBIV9BHF05bNLgOBk/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1kXDkdd0BNPoMZcMqcU-M0ZvwjlXvuiAB/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/11mivvvszQX6itp_tdRYXZ8hx9wHCU3TH/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 1 in args.levels:
            level = 1
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1MUk-vEtusn4-OHEk-pKxCkw6ktpY0lJV/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1tdh0LOnotC0--8gTXSAqOX4Nfs9pwVKl/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1LTFDA10nx_V315QkX_eMdeWFd0yE5qy3/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1hZYXSgiIB7kyFscEABXP1LpIqn7Pgh4L/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1NoMc5-0gnfGRT5BASZi2oLCGpa7iGcFx/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 2 in args.levels:
            level = 2
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1DRWTlMMkHXgGU3sFbAyB0FN8PZnSLcgS/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1JpYwr39VALZy1XKe9qslmY097TGz6zKx/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1nZnlOvhDpw3FqB_sXk2fcOtmRr_UfhMN/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1q2osu-teT6pOtMRDVWJOKeNryfKxyz-a/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1FV1OFpLQ_F8QsdKAd4c3SUn8OSFYP8Dq/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 3 in args.levels:
            level = 3
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1rNXn_vuhU9HCO59aLuG7CJktVGxXfj9s/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1uc4FW94q6wnRaAG66RSxd8L1wEh93kYM/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/17gDiQN3SQ26sQvduL1QZP9FcPc1HXwfh/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1yAov8iFF_KSvm9NJh_UcT7--jGZ0ijcl/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1Lco5D1uqNzgZs3vp2AYH4JPjV4evN60B/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)

    if 10 in args.scenes:
        scene = 10
        if 0 in args.levels:
            level = 0
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1Qg8coXZHOKNbEegADkedb9JVjgIpaHHZ/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1uHAHp1qcxBcHVQvHNttvH36REI_RxOM9/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1cMWiMKjSSeuX5OAejqxuUgSbTpMKRjoM/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/15S6RC76kkNOwIoFPvpkae8QJErdrR6ST/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1AD51p5lr33qXY8f7m1zP_amb7m6y_i2c/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 1 in args.levels:
            level = 1
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1eXzpysSlBa6-phsg9vhRzpQWA6rks4f_/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1HMOJS1aUmCFOmZta851V52ef0zoPZjLe/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1MtZ_cDjBYYKTCef6jx3pRGKHfC5bHZVG/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1TE0an-xytKJnc2b8lT7UABQ4QOXtbFCP/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/10oUF-Fjm6Dau0Ewwt3lC1B5kpjQvEfi1/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 2 in args.levels:
            level = 2
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1kny5MUusjxiErA7LGLDQ3fgm4UFSpPHj/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1-CmJPPKFRL-0rhfcDsBgsT_W8jRZ8_l4/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1pDiIoI7hf4sOknjwwfjW2lNJZ4xzGcXy/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/138BYUyCdUF8_gmmw0i2XRVMj1iCZIMbz/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1TkQ6ZZbuY4_Rx9Rml9fLheWtsVBmd0AV/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 3 in args.levels:
            level = 3
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1e3APO78N6V9ENR30LeGigIBJS60XJNHr/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1am6EKDZnEakteDwMMCIoc28AnSu2dTFX/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1ZVUrqRevzFz_mx71PcocmjTr7f3RHe99/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1mIZo2u1ztT61I3z6Go7r7CrxeFQGfIZW/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1d6birr-Y-uAhqD8g9_s4ukvlOI7sgjpE/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)

    if 11 in args.scenes:
        scene = 11
        if 0 in args.levels:
            level = 0
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/17Ums_SDL6MzRPSGW2sfTtAc42XO_95XB/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1yEHouK74IuqRz7xnXoaj_n3OcsDBkv_T/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1vdhlSiMl5bnwtOA1BieKQ2wJUHejhMUd/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1yws962BPCd14lyAF6ME_UNH0_qvl-8Fn/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/132QN4wfzibV4ea_r6R6hUWBovqoUIPLU/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 1 in args.levels:
            level = 1
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1ZZqJqG7EOkrNb28xCpyAkha1AZ0dT6cd/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1brXocb-gEFq1qppbQMOXTFvly8HNwFGd/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1lI8ELqxyGa4ICDpmpmGj492K9KbrnjhH/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/17mzkgH2CsbHOu9i6txiUNWpjUQfFRyHm/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/16d6KdufS9g_9gLRwrMRQgfIgjXeNFtal/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 2 in args.levels:
            level = 2
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1AaAaFtCKQF5ugjzH5W5lWqRij-Dk9spH/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1bOMz0-t3DB8YLpyKEb0LSJ0HIFg6q_VQ/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/135TihjuforOYxzcn5BImlmjajW8nbkoI/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1dGx1IniOr8KwVUmTZbgbZMKPIwqXyDhv/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1hwnCoeScLKPUTRoCwVzw5GsBDa0JSOfJ/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 3 in args.levels:
            level = 3
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1ExmjH9_XsO9luMz9FTEyHXHGnTqbeul7/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1Fj-jwkfYraShcz_kK-hKvqoOSZlJnbVE/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1nGgJ4drsO0Hm9mKXRceEsO0O1KQJ4c8G/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1I5_AO8Ag7KPDBnD0rzARueZ-HEfN8ZqV/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1TKtEoMyt-w7VrsBTRFMsZQZcoNCCQD56/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)

    if 12 in args.scenes:
        scene = 12
        if 0 in args.levels:
            level = 0
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1wGjqT3gDbSctYJ3OXKUxINcLwkIP9FH9/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1DBp37BAWc3s6tWKUmvmR1FT-sH6vdjee/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/10Yjyxg2TIWZIoVDQOi5TYWTkJpQ04bQL/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1jP4smxJ27opGy5ICjZhYGiX3dQv2L7tI/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1S8foH2afUQX5lI1duGfZwayRTEVGKqRd/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 1 in args.levels:
            level = 1
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1F8PqpKHkhoVBJ9QZRqIOxe4AKLGzjlHG/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1ZdlcTK08gmbyBKmLpxAKLXLGoXOS8edm/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1KU41R6T0EExlsl-rvv3Hi146GzC9NPoN/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1TuHpobNkYplvVAlHJf-V70v-2PVHaEVL/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1bk1gEFJ0PjF5acYr0X69FQKSzcK1x_zh/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 2 in args.levels:
            level = 2
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/18ymkVJROiL41nlYlm4ccNToByg5EZZat/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/13oiNpoExQCHjf1A4tt7xHM3n2oU1Ajwi/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/15RqYekHEjUebeEuvnQI6ew3NdgjNndcu/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1xqVJ2g1CohiS5BjDMdinf3DRPBlJY5Lw/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1tRMwn68cESB3TrmOkPVX4qL9hfhWw227/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 3 in args.levels:
            level = 3
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1bCW0K9zuJLDbFBqp4TqKlEBq7fUceEUr/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1uD-9l3l1g4iwmDUN1nZGjkT-pkWJzwq4/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1VeL5sOdXNWfNjCp3EAefY6VK9BdTFzfA/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1eXowlKkjlr7PqTusrVPOVt1sRNNPvs5P/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1t6u49dDj0UoEQ8LADJhC8rHJnrSrQgZl/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)

    if 13 in args.scenes:
        scene = 13
        if 0 in args.levels:
            level = 0
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1WJtF_Gi61O8BC8lkssrjZyul9vakwQgG/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1tWnLbSynenDeNBJ5meDsS3GT6wUXh6PM/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1nGmhikexXvyYX61h3IbawYiRjJ6SwwJJ/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1eHMHlbKL3O8GQrLin0ZCTpsgkbF_Enx0/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1eABEzgNsz1_b3tYL3e-9zkGMbP58OVxW/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 1 in args.levels:
            level = 1
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1756D3ngzpc32wzH0oAFK0Sq6eBDuZYIT/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1-TWKV4LPF9aK5M9YhEmFGhKtsBqLq-0V/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1PKRA4tTIhgWlc7c6dhAav6_RfrpMQbmB/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/10EVGoSXVt71v_ovQOs1ecWyJLgeVg4XE/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1--4JzHURvt4RvbRqGWOrUvAU0eu7JQRE/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 2 in args.levels:
            level = 2
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/17l70iviA7UmrbKEXVl6P7BtA2lleFNx5/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1ZmuKPtPOgS5wLHwirhBOXQBZ52kNMdfQ/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1eUjPJC-SA-OcDwWCIu68yhn7RiAooBgV/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1DkLp9qKyrQ1iT3-F_e6Gdi6NUh_PnkzJ/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1avbQAQ-Oal0BE8kc1602IGVFHZccMTMr/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 3 in args.levels:
            level = 3
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1GASQ205qHUKtZPh-DTkf37bDbkRCyltO/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/13bpJp6_OcqJgZ6t2WiOF1uc9U4ysYBU2/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1nrDejj87h1B2EnL28WMc5D78ufEMCAvX/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1BMUi9Ut9pU1QqSule-ExxH4bjrzXU3pg/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/17sDu_3dMG4IswDXeNVbb-yG7f6ekxSEh/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)

    if 14 in args.scenes:
        scene = 14
        if 0 in args.levels:
            level = 0
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1FXftKsKRnqDrMRSCiGlwhjl_AorT7k36/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1Du7PU13qYj2Idl9WHmirO--wSri7A4Dw/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1y7mXxAuSThcGHuyRySjxns1Jj01lSxwG/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1pAzTvgaH1VXL5adZ_ZSzn-eTuyDLVZF0/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1RG3Ryl3QTL-QZ12_TNHb4WP0U-x7gLNJ/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 1 in args.levels:
            level = 1
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1e3L8RPAWy2so3C8nazOf2CrjE_IxCLas/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1tu9106iTQKociJll_X7D4UrZo0-97dqq/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1wguOzFHNjVr896ENm4rbaekRUk1lILe0/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1iZEcsMuiwtD7hVS9aj7Ek_TUB_oY_6GT/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1f9_c6YxN6SyjUvu1IribSQvxrbTO5D7N/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 2 in args.levels:
            level = 2
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1Aacmg6KMrRxGkVDDqCxL3RkowXDDw37b/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1e5yfaMOtFoB8mrJkoEsszR6LRJKSTJNq/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1nhZIzSFzZYWP0O_HjLMF_kGBFNYOy0W2/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1RLYhnCJC1D_7p09P5pK-9rExAwdqXmHY/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1JywLm3Jm4icbiyppqNf-V7n4MJZflJl4/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 3 in args.levels:
            level = 3
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1AIpm8DIrdcmsLjtIT2YR51GglpwR3q0p/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1t186ZpL6IRDKEsL751M75-AjFP4iHluY/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1cQkw-i0COK_Pk6bgvMmgL-X_7o94YjBc/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1pf5EHmIrULcAMVUSE2_KfA5Ym2Qsb_S2/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1pA1_M_SFFkTvNoYdI1Z1lw77k5upq9g6/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)

    if 15 in args.scenes:
        scene = 15
        if 0 in args.levels:
            level = 0
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1ORmFB7aWsP43DElwrj2tMlyBPpOgNL9c/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1ekKScZlo5Euck8l3SGnOTCVNekEAi9A8/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1to2P24NgBxrrjZHITRbOkdUB4UmlLxNa/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1xCOvZ6FBCF2Pdx3CtfKeCmFmXajuO7G3/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1lpHb078YdtTOyolzZHWPqwSzcXq7vAYd/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 1 in args.levels:
            level = 1
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1Qn0pYSA_Jxrjy4Bdy1Ob4fNA0gmTvKYD/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1kZp4MJ7hezuBPErLGE3eVT0MVOfNxyvC/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1jbCZnbqQw78cnqiOERRekn8BYd5FnY66/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1DUB4DoMwhaF3Q57mlKQ7oUrp5WksgFmX/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1l6oEgFIJgCOkkcFkfWf98PX9gOs61kfv/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 2 in args.levels:
            level = 2
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1qaoglU96dWo4k8fY_nkaFqEAuW4BSMa3/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1DPlnRccrb67WCBjHyWzv0vgvAHaWEYIB/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1A-v4K-tBrk62TH0YNW068Rqk6SO_xcie/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1HD9siu6Tw7HUWLdq-v0I-1yyoGR4tmHt/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1tY410DawAc83pbXHcroAfZPY8-O-PBJ3/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 3 in args.levels:
            level = 3
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1NJK1Z1ebLcxCt8QwUhJcj9Jx7zXhHkVQ/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1dGsuOwHjlBGy8BlSKSN5xOmiYMgUpDlC/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1TXFOC-A0lNopKuXqVmOAU5Fcdp22d06j/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1M5iyAVXruo9S0Tct441Lqwga2sq0Y1BA/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1tqOOZwCkSEoIOr9n1U7XGg3aawOVvrIn/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)

    if 16 in args.scenes:
        scene = 16
        if 0 in args.levels:
            level = 0
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1BkY36OIxMaqfoD4kGYlio0U6fYo2b09b/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1ZQKLit5nlS9hAuCP692yVYfuD7px65_e/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1mNCrUyHVcvN791KHyj-R4K6LTDXHqHGy/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1rb8F9bxoPaZ9qm9NIeCD3bI-6ab2mcEP/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1a0f_ihNi9jHX4lF8t8UN1RroA-PyWOgw/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 1 in args.levels:
            level = 1
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/16704qnE8uPxdmjAX6nsI2szkXP2EaaFW/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1hw3lsIYsOLMFYboOL8tp4eRjrs4L0V8q/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1RHW44djXRQFUA9CgF_hw0QQvF5pBKhJB/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1kmzhG2d2I9T6r1G8VJ3bWa1bMtkrWgLa/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/157NJlUhaOTCl-akihQLZWCPCMjssOMbX/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 2 in args.levels:
            level = 2
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1XZHQcNLodULqll02MUumxKAGUGPnCDeI/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1HUsBIoDqgWy5QW9AhRdxqB7FktlVNNJz/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1eGS7ETJ5l5uODDBnfCpz4Wir66dwtr_Z/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1pgS3GoQhUsOxCcPnigcOcqxfGSdrARvP/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1XrmzNO2RAMQMxHtIQmX_qvPK71SY_yvN/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 3 in args.levels:
            level = 3
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1KA32i8BndCnvCghAnyk-jNb5rF3MGqci/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1EhkNTb78Cu9V_XelxUiE568x5EFrldwy/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1vhpel8Ya-bbqDEBhQTbq7vpZJG7hjuZ5/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1Ik2tYV_pUhrc5xHcQ59Qk0rygbvI-vxU/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1-Gj4lz6Qea8jKTkb3S_9Ekk06yhAUKnM/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)

    if 18 in args.scenes:
        scene = 18
        if 0 in args.levels:
            level = 0
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1xzeubGpm8lH8hYkhPioHrjy6Qj7trNrq/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1V74zyb1MvgGNGWMawECPTdwUjeoMRaao/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1CJYLXl3uiKjPACvGTmO-aikbwgwJzypc/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1rmnDB1wteS9HcCRi0LAcK753xcwJQlN_/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1W4mCK5-XjCdF2HlKviaPHuMDf_3HgjWO/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 1 in args.levels:
            level = 1
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1AdwbIuDv3o6eMyONLqbpuIftbkH73l1T/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1ssMjGmHtrT6d-hWL6Q9PihZ0h5GxR3-M/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1oXZPAy_rQt5-GwMMw8joEh5dsPu2Q-LV/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/15pymbrWUhgxsHXfkOlaHt35aThoMyOpl/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/13laVCXO9ckXnDq1yW0b424JVUMAA0OUW/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 2 in args.levels:
            level = 2
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1NVL3OptyAYrfnzsIoaVZ-5jG0S1kwvoH/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1uVmjjD-4OvtQQs9eaMh-mIB71_d1P0Ki/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1obIAkeDkYDjbAHbQbabHfpQsJFl_DaKa/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1vkwMpDZJqfkWejzucg6-38LQ5arwU4C8/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1iAg7f8-eTeesRqcMGg71vmWN8rtWq5y5/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 3 in args.levels:
            level = 3
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1_OBa7QU8g3tn9JYpnrzpZ158Y_QESoDE/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1bsQ9vX6N6uMMVkigxqWBpMICzsYk1zLK/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1sjuAwN6RzNnDgEhhTmqVmdfhKlhmMLv2/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1Ova-ROqTX8z2PqQkxUmKdpV1eI1T_8pH/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1aZ9FKgB5YE5RgPhnub6aqHGgCa8J7erl/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)

    if 19 in args.scenes:
        scene = 19
        if 0 in args.levels:
            level = 0
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1ECmMO9s2RiotftCeUfJQCxS1pbhhBRII/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1Sm0fsnQ2avC4MYeqtkHOt-IQR2Rujm34/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1THvOi7SphyicUqNs2IWHg2wIPs6tVkqL/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1boHOBEe5z2qdQQ6rdCQ967uUReF1mvbn/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/17nJtSroAzb_-0ijL7AmzSbXYRuI3sxw5/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 1 in args.levels:
            level = 1
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1X32XioczPjT5Wzr1s6gxUP03EjCKjaDk/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/14QNkDHbuIcGweikjCCH-Yi4f-Vw7Rzhe/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1dHj4LdOkDR0wjzyN_JDJRlC_S5uQz2aU/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1qpoar8nLHoTI3CcU3dlKteMVkgYwIyJs/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1lnyqMEUC8P0NSMVrR4t7Yx5jpLgGkmNd/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 2 in args.levels:
            level = 2
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1s3UbaboQ0mT1jpKyNmKrmV5PV1l0OAQ4/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1aKvvwv9hO2KtW-G2j6Yo5SsSDwZjuhFH/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1LDOSObr-xXddyW-ticUlFwrxfKGz75Sa/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1nPLba0R4oCEQ9ZoyYbbFEIhMBZJpxGab/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1MOA4IuBSU5s8aWeMW6J76oTqS1qrnWmV/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
        if 3 in args.levels:
            level = 3
            output_folder = os.path.join(args.output_folder, '{:05d}'.format(scene), 'level{}/'.format(level))
            if args.color and not os.path.exists(os.path.join(output_folder, 'color.zip')):    
                print("Downloading: scene {} - level {} - color.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1pB9T1HOK2wJZJIJUSCt9dIfqPF31Vko7/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.depth and not os.path.exists(os.path.join(output_folder, 'depth.zip')):
                print("Downloading: scene {} - level {} - depth.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1Y20twRgOjtpu4Ba8c2_V733PGfY6w0ae/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.optical_flow and not os.path.exists(os.path.join(output_folder, 'optical_flow.zip')):
                print("Downloading: scene {} - level {} - optical_flow.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1-KSzKwwBZsWP64rNYxhLWXUYdpRQUFoQ/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.normal and not os.path.exists(os.path.join(output_folder, 'normal.zip')):
                print("Downloading: scene {} - level {} - normal.zip".format(scene, level))
                url = "https://drive.google.com/file/d/1EWNxuPmvnE5ZzoQRRqxK8cBzLUJ52XzH/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)
            if args.pose and not os.path.exists(os.path.join(output_folder, 'pose.txt')):
                print("Downloading: scene {} - level {} - pose.txt".format(scene, level))
                url = "https://drive.google.com/file/d/1FpBYMctdFM8P5q3uMPb-GPYwvxOe2g3e/view?usp=sharing"
                gdown.download(url, output_folder, quiet=False, fuzzy=True)

    if args.unzip:
        zip_files = []
        start_dir = "/media/david/DiscoDuroLinux/Datasets/Matterport/train/poses"
        pattern = "*.zip"

        for zip_file_path, _, _ in os.walk(args.output_folder):
            zip_files.extend(glob(os.path.join(zip_file_path, pattern)))

        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_file = Path(zip_file)
                zip_ref.extractall(zip_file.parent.absolute())
            os.remove(zip_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--color', action='store_true', help='download color images')
    parser.add_argument('-d', '--depth', action='store_true', help='download depth images')
    parser.add_argument('-f', '--optical_flow', action='store_true', help='download optical flow maps')
    parser.add_argument('-n', '--normal', action='store_true', help='download normal maps')
    parser.add_argument('-p', '--pose', action='store_true', help='download camera poses')
    parser.add_argument('-l', '--levels', nargs='+', choices=[0, 1, 2, 3], type=int, help='levels of difficulty to download')
    parser.add_argument('-s', '--scenes', nargs='+', choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19], type=int, help='scenes to download')
    parser.add_argument('-o', '--output_folder', type=str, help='full output folder path where to download the dataset. By default is downloaded in the folder where it is executed', default=os.getcwd())
    parser.add_argument('-z', '--unzip', action='store_true', help='unzip downloaded files and remove all .zip files')
    parser.add_argument('-a', '--all', action='store_true', help='download all data. If specific data is given in arguments (e.g., --color), only that data will be downloaded for all the scenes.')
    args = parser.parse_args()

    download(args)
