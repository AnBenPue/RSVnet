# PointNet

Implementation of a PointNet model:

* [PointNet](https://github.com/charlesq34/pointnet): Deep Learning on Point Sets for 3D Classification and Segmentation. Created by Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas from Stanford University.

* Based on the implementation of [garyli1019](https://github.com/garyli1019/pointnet-keras).

## Project tree

    . PointNet
    |── data              # Dataset
    |── models            # Pretrained models

## Usage

Make sure that the *.h5* files containing the data are in the correct folder. In order to train the model run:

```bash
    python train.py
```

The model will be saved with the name *model_saved.h5* in the models folder.  The model evaluation can be done running the following file:

```bash
    python test.py
```

In order to evaluate the model using a pointcloud, run:

```bash
    python test_scene.py
```
