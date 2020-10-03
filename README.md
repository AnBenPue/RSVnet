# RSVnet

The RSVnet model combines the PointNet model with the voting and clustering process described in [Rotational Subgroup Voting and Pose Clustering for Robust 3D Object Recognition](http://openaccess.thecvf.com/content_ICCV_2017/papers/Buch_Rotational_Subgroup_Voting_ICCV_2017_paper.pdf).



## Installation

Run the configuration script in order to create a virtual enviroment for the repository and install all the necessary packages.

```bash
chmod +x config.sh
./config.sh
```

Add the repository path to the *PYTHONPATH* environment variable.
Edit your *.profile* and add the following lines:

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

## Dependencies

* [tensorflow](https://www.tensorflow.org/)
* [keras](https://keras.io/)
* [h5py](https://www.h5py.org/)
* [matplotlib](https://matplotlib.org/)
* [Pillow](https://python-pillow.org/)
* [open3d](open3d.org/)
* [sklearn](https://scikit-learn.org/stable/index.html/)

### Related Projects

* <a href="http://openaccess.thecvf.com/content_ICCV_2017/papers/Buch_Rotational_Subgroup_Voting_ICCV_2017_paper.pdf" target="_blank">
  Rotational Subgroup Voting and Pose Clustering for Robust 3D Object Recognition</a>
  by Anders Glent Buch,  Lilita Kiforenko,  Dirk Kraft.
  
* <a href="http://stanford.edu/~rqi/pointnet" target="_blank">
  PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</a> by Qi et al. (CVPR 2017 Oral Presentation).
  Code and data released in <a href="https://github.com/charlesq34/pointnet">GitHub</a>

* <a href="https://github.com/garyli1019/pointnet-keras" target="_blank">
  PointNe-Keras: Keras implementation for Pointnet </a>
  by  garyli1019 .
