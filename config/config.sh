#!/bin/bash

# Install virtual environment
sudo pip3 install virtualenv
sudo apt-get install python3-venv

# Create necessary folders
mkdir ../Linemod/ply
mkdir ../Linemod/pose
mkdir ../Linemod/object_classification/data
mkdir ../Linemod/RSVnet/data
mkdir ../PointNet/object_classification/data
mkdir ../PointNet/object_classification/models
mkdir ../PointNet/scene_classification/data
mkdir ../PointNet/scene_classification/models
mkdir ../RSVnet/data/eval
mkdir ../RSVnet/data/test
mkdir ../RSVnet/data/train
mkdir ../RSVnet/models

# Create a new virtual environment
python3 -m venv ../.venv
source ../.venv/bin/activate

# Install all the necessary packages
pip3 install -r requirements.txt

