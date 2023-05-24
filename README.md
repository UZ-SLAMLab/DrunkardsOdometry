# The Drunkard’s Odometry: Frame-to-frame Camera Motion in Deforming Scenes

<center><img src="assets/Overview_drunk.jpg" width="540" style="center"></center>

[The Drunkard’s Odometry: Frame-to-frame Camera Motion in Deforming Scenes]()  
David Recasens, Martin R. Oswald, Javier Civera


## 💭 About
The Drunkard’s Odometry is a robust flow-based odometry method. 
The Drunkard’s Dataset is a challenging collection of synthetic data targeting visual navigation and reconstruction in deformable environments.


## ⚙️ Setup

We have ran our experiments under CUDA 9.1.85, CuDNN 7.6.5 and Ubuntu 18.04. We recommend create a virtual environment with Python 3.7 using [Anaconda](https://www.anaconda.com/download/) `conda create -n edam python=3.7` and install the dependencies as
```shell
pip3 install -r path/to/DrunkardsOdometry/requirements.txt
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


## :beers: Drunkard's Dataset Evaluation


## :man_health_worker: Hamlyn Evaluation


## 	:call_me_hand: Demo

You can run the demo to predict the relative camera pose from a pair of RGB-D frames:

```
python scripts/demo.py ...
```
