# The Drunkard’s Odometry: Frame-to-frame Camera Motion in Deforming Scenes

<center><img src="assets/Overview_drunk.jpg" width="540" style="center"></center>

[The Drunkard’s Odometry: Estimating Camera Motion in Deforming Scenes]()  
David Recasens, Martin R. Oswald, Javier Civera


## 💭 About
The Drunkard’s Odometry is a robust flow-based odometry method. 
The Drunkard’s Dataset is a challenging collection of synthetic data targeting visual navigation and reconstruction in deformable environments.


## ⚙️ Setup

We have ran our experiments under CUDA 9.1.85, CuDNN 7.6.5 and Ubuntu 18.04 and (check eth cluster). We recommend create a virtual environment with Python 3.7 using [Anaconda](https://www.anaconda.com/download/) `conda create -n edam python=3.7` and install the dependencies as
```shell
conda create --name drunkard --file requirements.txt
```


## 💾 Data

The Drunkard's Dataset can be found [here]() and The Drunkard's Odometry models [here]().

Expected directory structure:
```Shell
├── drunkards_dataset
    ├── 00000
        ├── level0
            ├── color
            ├── depth
            ├── optical_flow
            ├── pose.txt            
```


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

To run the Drunkard's Odometry on all the four levels of difficulty of the Drunkard's Dataset test scenes:

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


## 	:call_me_hand: Demo

You can run the demo to predict the relative camera pose from a pair of RGB-D frames:

```
python scripts/demo.py ...
```
