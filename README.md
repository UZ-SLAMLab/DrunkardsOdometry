# The Drunkard’s Odometry: Estimating Camera Motion in Deforming Scenes

<center><img src="assets/Overview_drunk.jpg" width="540" style="center"></center>

[The Drunkard’s Odometry: Estimating Camera Motion in Deforming Scenes]()  
David Recasens, Martin R. Oswald, Marc Pollefeys, Javier Civera


## 💭 About
This repository is the official implementation of The Drunkard’s Odometry, a robust flow-based odometry estimation method, and contains information about the Drunkard’s Dataset, a challenging collection of synthetic data targeting visual navigation and reconstruction in deformable environments.


## ⚙️ Setup

We ran our experiments under CUDA 9.1.85, CuDNN 7.6.5 and Ubuntu 18.04 and (check eth cluster), using a single RTX Nvidia Titan GPU during training and/or a single RTX Nvidia 2080 Ti for evaluation. We recommend create a virtual environment with Python 3.7 using [Anaconda](https://www.anaconda.com/download/) `conda create -n edam python=3.7` and install the dependencies as
```shell
conda create --name drunkard --file requirements.txt
```


## 💾 Data

The Drunkard's Dataset can be found [here](https://drive.google.com/drive/folders/1AZHUKMbe7bR1xwRmAAZ0AHgcEqRnNjms?usp=sharing).

Expected directory structure:
```Shell
├── drunkards_dataset
    ├── 00000
        ├── level0
            ├── color
            ├── depth
            ├── optical_flow
            ├── normal
            ├── pose.txt 
            ├── wrong_poses.txt (few times)          
```

For every of the 19 scenes there are 4 levels of deformation difficulty and inside each of them you can find color and depth images, optical flow and normal maps and the camera trajectory.

- Color: RGB uint8 .png images. 
- Depth: uint16 .png grayscale images whose pixel values must be multiplied by (2 ** 16 - 1) * 30 to obtain metric scale in meters.
- Optical flow: .npy image numpy arrays that are .npz compressed. They have two channels: horizontal and vertical pixel translation to go from current frame to the next one.
- Normal: .npy image numpy arrays that are .npz compressed. There are three channels: x, y and z to represent the normal vector to the surface where the pixel falls.
- Camera trajectory pose: .txt file containing at each line a different SE(3) world-to-camera transformation for every frame. Format: timestamp, translation (tx, ty, tz), quaternions (qx, qy, qz, qw).
- Wrong camera poses: .txt file containing corrupted frames and the immediately adjacent ones that are rejected in the dataloader. It barely happens for some specific cases, not in the used test scenes (0, 4 and 5). It is being currently addressed.

Check the Drunkard's Odometry dataloader for further coding technical details to work with the data.

## 🧠 Training

To execute a small training test over the [Drunkard's Dataset Sample]():

```shell
python scripts/train.py --name=drunkards-odometry-test --datapath=/.../DrunkardsDatasetSample --difficulty_level=1 --batch_size=2 --train_scenes 0 --val_scenes 0 
```

To replicate the paper training:

```shell
python scripts/train.py --name=drunkards-odometry --datapath=/.../DrunkardsDataset320 --difficulty_level=1 --depth_augmentor
```

For a personalized training you can play with the different arguments:
- --name: name your experiment.
- --ckpt: .pth model checkpoint to load.
- --continue_training_from_ckpt: continue training from loaded model. Total steps, optimizer, scheduler, loss, clipper and number of trained epochs are restored from checkpoint. Otherwise, training starts from zero, starting from the pretrained loaded model.


## :beers: Drunkard's Dataset Evaluation

To run the used [Drunkard's Odometry model](https://drive.google.com/file/d/1ZQhr3iQobRaeofNaeCNu2e1MGA1PrnCJ/view?usp=sharing) on all the four levels of difficulty of the Drunkard's Dataset test scenes:

```shell
sh  scripts/eval_drunkards_dataset/run_drunkards_odometry.sh
```

You need to modify the arguments to specify your dataset's and model checkpoint's path. By default the evaluations are saved in the folder 'evaluations_drunkards_dataset'. This script outputs the estimated camera poses 'pose_est.txt' and the flow metrics 'metrics.txt'. There are also pose estimation metrics without trajectory alignment, used in the training ablation study.

To obtain the trajectory metrics (RPE traslation, RPE rotation and ATE) you need to execute this script to align the estimated and ground truth trajectories before the evaluation. 

```shell
sh  scripts/eval_drunkards_dataset/drunkards_odometry.sh
```

Beforehand, modify the evaluations root path in the script and place the ground truth poses "pose.txt" in the same folder as the estimated ones "pose_est.txt" following this structure:

```Shell
├── evaluations_drunkards_dataset
    ├── 00000
        ├── level0
            ├── drunkards-odometry
                ├── pose.txt
                ├── pose_est.txt                       
```

To evaluate the trajectories of the other baselines (Colmap, DROID-SLAM and Endo-Depth-and-Motion) use the following scripts

```shell
sh  scripts/eval_drunkards_dataset/colmap.sh
sh  scripts/eval_drunkards_dataset/droidslam.sh
sh  scripts/eval_drunkards_dataset/edam.sh
```

You need to save the estimated poses 'pose_est.txt' with the same folder structure as the Drunkard's Odometry, but name the method's folder as 'colmap', 'droidslam' or 'edam', respectively. In addition, in those cases where not the full scene has been tracked, you must remove those missing poses from the ground truth file 'pose.txt'.

## :man_health_worker: Hamlyn Evaluation

As the Hamlyn dataset does not have ground truth camera poses, to evaluate the trajectory estimation we introduce ground truth-free metric Absolut Palindrome Trajectory Error (APTE), that averages the L2 error between start and end pose of the estimated trajectory for the palindrome video (k-frames forward and backward) over all possible loop lengths. Therefore, this metric needs the forward and backward estimated trajectories.
For the evaluation, we use the same [data](https://drive.google.com/file/d/1Iqdk8P51FuD5O96mO_8YubyKQ6BzQRp9/view?usp=sharing) as in the tracking experiments of the paper [Endo-Depth-and-Motion](https://ieeexplore.ieee.org/abstract/document/9478277?casa_token=jo2SHKzVwd8AAAAA:EEsxN4CvnZr8BcASDFF5GdXIqVX7cWGiYUIyhuQ19iz4GF7vsK1f-GkfHRhsh0hmEtdb__aVDg), that consists in forward and backward RGB images of test scenes 1 and 17 with the black borders cropped out and the estimated depth maps by a single-view self-supervised network with the following structure:

```Shell
├── HamlynData
    ├── test1 (forward)
        ├── color
        ├── depth
        ├── intrinsics.txt
        ├── intrinsics_matrix.txt     
    ├── test1_backward                  
```

To run the used Drunkard's Odometry models that were trained [with](https://drive.google.com/file/d/1kmg4D9q8X3pYhdpPNfCb_WYSBrfnKkCM/view?usp=sharing) and [without](https://drive.google.com/file/d/1OUcpYJTP_5rXdadeK6wEVi2IZmBiOctX/view?usp=sharing) deformation:

```shell
sh  scripts/eval_hamlyn/drunkards_odometry.sh
```



## 	:call_me_hand: Demo

You can run the demo to predict the camera trajectory from RGB-D frames:

```shell
python scripts/demo.py --name=demo --ckpt=/.../drunkards-odometry.pth --datapath=/.../DrunkardsDatasetSample --intrinsics 190.68059285 286.02088928 160. 160. --depth_factor=0.000457771 --depth_limit_bottom=0.1 --depth_limit_top=30.
```

In this example, we are estimating the pose on Drunkard's Dataset samples, thus substitute them with your own data and the parameters accordingly.


## 👩‍⚖️ License

The code, dataset and additional resources of this work are released under [MIT License](LICENSE). There are some parts of the code modified from other repositories subject also to their own license:

- The code in drunkards_odometry and data_readers folders is based and extended from [RAFT-3D](https://github.com/princeton-vl/RAFT-3D) under [BSD 3-Clause License](https://github.com/princeton-vl/RAFT-3D/blob/master/LICENSE).
- The code in drunkards_odometry/pose_cnn is derived from [Manydepth](https://github.com/nianticlabs/manydepth) under [ManyDepth License](https://github.com/nianticlabs/manydepth/blob/master/LICENSE).
