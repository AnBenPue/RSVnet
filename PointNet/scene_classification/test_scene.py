import os
import sys

import open3d as o3d

from cloud_splitter import cloud_splitter
from configuration import file_paths, var_scene_classification
from PointNet_model import scene_classifier

# Configuration parameters
v = var_scene_classification
f = file_paths

# Declare the model
scene_class = scene_classifier(v.INPUT_SHAPE, v.NUM_OF_CATEGORIES)
# Load the scene point cloud
scene = o3d.io.read_point_cloud(f.SCENE)
# Split the scene cloud in segments with the correct size for the pointnet
# module
cs = cloud_splitter(scene, v.INPUT_SHAPE)
seeds = cs.getSeeds(v.VOXEL_SIZE_SCENE, v.VOXEL_SIZE_SEEDS)
segments_data = cs.getSegments(seeds, v.SEGMENT_RADIUS)
# If  no segments are found, stopt the execution of the script.
if segments_data is None:
    sys.exit()
# Unpack the segments data
[X_test, X_test_u, X_normals, _] = segments_data
# Load the pretrained model
scene_class.loadModel(f.SCN_CLASS_POINTNET_LOAD)
# Run the pointnet in order to classify the segments between object and
# background
scores = scene_class.predict(X_test)
scene_class.visualizePrediction(X_test_u, scores)
# From the segments classified as objects, select only the best candidates
c_data = scene_class.getCandidates(scores, X_test_u, X_normals, num_of_candidates=1)
# Unpack candidate data
[c_points, _, _] = c_data
scene_class.visualizeBestCandidates(c_points)
