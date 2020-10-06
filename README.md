# RSVnet

This pose estimation method combines the PointNet model with the voting and clustering process described in [Rotational Subgroup Voting and Pose Clustering for Robust 3D Object Recognition](http://openaccess.thecvf.com/content_ICCV_2017/papers/Buch_Rotational_Subgroup_Voting_ICCV_2017_paper.pdf).

This project was part of my master thesis, for a deeper explanation of the pose estimation process, the report can be found [**here**](https://drive.google.com/file/d/19ugY-eDmjGruppoCufYvgZsz8mI4UjDE/view?usp=sharing).

## Installation

Run the configuration script in order to create a virtual environment for the repository, install all the necessary packages and create the necessary folders.

```bash
cd ./config
chmod +x config.sh
./config.sh
```

Add the repository path to the *PYTHONPATH* environment variable.
Edit your *.profile* and add the following lines:

```bash
  gedit ~/.profile
```

```bash
# set PATH for the RSVnet repository
export RSV_REPOSITORY_PATH=/path/to/the/repository
if [ -d  "$RSV_REPOSITORY_PATH/" ]; then
    export PYTHONPATH=$PYTHONPATH:"$RSV_REPOSITORY_PATH/Linemod/object_classification"
    export PYTHONPATH=$PYTHONPATH:"$RSV_REPOSITORY_PATH/config"
    export PYTHONPATH=$PYTHONPATH:"$RSV_REPOSITORY_PATH/utilities"
    export PYTHONPATH=$PYTHONPATH:"$RSV_REPOSITORY_PATH/PointNet"
    export PYTHONPATH=$PYTHONPATH:"$RSV_REPOSITORY_PATH/RSV"
    export PYTHONPATH=$PYTHONPATH:"$RSV_REPOSITORY_PATH/RSVnet"
fi
```

Source the *.profile* to update the changes.

```bash
source ~/.profile
```

This project was build using **Ubuntu 18.04** and  also requires the installation of CUDA in order to use tensorflow-gpu.  I followed these tutorial in order to install it:

* [How to install CUDA 9.2 on Ubuntu 18.04](https://www.pugetsystems.com/labs/hpc/How-to-install-CUDA-9-2-on-Ubuntu-18-04-1184/)

* [How To Install CUDA 10 (together with 9.2) on Ubuntu 18.04](https://www.pugetsystems.com/labs/hpc/How-To-Install-CUDA-10-together-with-9-2-on-Ubuntu-18-04-with-support-for-NVIDIA-20XX-Turing-GPUs-1236/)

Make sure that the paths defined in *config/configuration.py* are all correct.

## Usage
The first step is to add the [Linemod](http://campar.in.tum.de/Main/StefanHinterstoisser) data to the correct folder. The pointclouds should be in **.ply** format and the object of interest pose in a **.txt** file. 
The pointclouds should be placed in the Linemod/ply whereas the pose files go to Linemod/pose. 

Once the data is added, the following step consist on generating the dataset for training the different PointNet models and the regression model. To do so, we need to run: 
```sh
python Linemod/object_classification/main.py 
python Linemod/RSVnet/main.py 
```
As a result each script generates a set of **.h5** files with the data necessary for training and evaluating the models. The data can be found in:

* Linemod/object_classification/data 
* Linemod/RSVnet/data

Regarding the data for the scene classification model, it can be downloaded from **[here](https://drive.google.com/file/d/1Sw7GWFoa42TzzC4roF6dawRPTY-nc_4b/view?usp=sharing)**. The process for obtaining the dataset is not implemented in this project. However the procedure is explained in detail in the master thesis report.

The dataset files can be split in different groups for training, test and evaluation. Each model has its own data folder: 

* RSVnet/data
* PointNet/object_classification/data
* PointNet/scene_classification/data

In order to train the classification models, we can run:

```sh
 python PointNet/scene_classification/train.py 
 python PointNet/object_classification/train.py 
```

The trained model will be saved into:

* PointNet/scene_classification/models
* PointNet/object_classification/models

In order to evaluate the classification models, we can run:

```sh
 python PointNet/scene_classification/test.py 
 python PointNet/object_classification/test.py 
```

The regression model can be trained by running:

```sh
 python RSVnet/train.py 
```

And evaluated with:
```sh
 python RSVnet/test.py 
```

Once all the models have been trained, the main pose estimation process can be run.

```sh
 python src/main.py 
```
## Dependencies

This project has the following main python dependencies:

* [tensorflow](https://www.tensorflow.org/)
* [keras](https://keras.io/)
* [h5py](https://www.h5py.org/)
* [matplotlib](https://matplotlib.org/)
* [Pillow](https://python-pillow.org/)
* [open3d](open3d.org/)
* [sklearn](https://scikit-learn.org/stable/index.html/)

## Related Projects

* <a href="http://openaccess.thecvf.com/content_ICCV_2017/papers/Buch_Rotational_Subgroup_Voting_ICCV_2017_paper.pdf" target="_blank">
  Rotational Subgroup Voting and Pose Clustering for Robust 3D Object Recognition</a>
  by Anders Glent Buch,  Lilita Kiforenko,  Dirk Kraft.
  
* <a href="http://stanford.edu/~rqi/pointnet" target="_blank">
  PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</a> by Qi et al. (CVPR 2017 Oral Presentation).
  Code and data released in <a href="https://github.com/charlesq34/pointnet">GitHub</a>

* <a href="https://github.com/garyli1019/pointnet-keras" target="_blank">
  PointNe-Keras: Keras implementation for Pointnet </a>
  by  garyli1019 .
