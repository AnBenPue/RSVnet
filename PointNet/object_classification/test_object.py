import os
import sys

import numpy as np
import open3d as o3d

from cloud_splitter import cloud_splitter
from configuration import file_paths, var_object_classification
from pointcloud import downsample, fromNumpyArray
from PointNet_model import object_classifier
from visualizer import visualizer

# Configuration parameters
oc = var_object_classification
f = file_paths

# Load the object seeds point cloud
seeds = o3d.io.read_point_cloud(f.OBJECT_SEEDS)
# Load the scene point cloud 
scene = o3d.io.read_point_cloud(f.SCENE_SEGMENT)
leaf_size = 6
# Center the cloud in the origin
scene = downsample(scene, leaf_size)
scene.translate([100,100,100])
scene.paint_uniform_color([0,1,0])
o3d.visualization.draw_geometries([scene])
# Split the scene cloud in segments with the correct size for the pointnet
# module. We use all the scene points as seed since we want a segment per point
cs = cloud_splitter(scene, oc.INPUT_SHAPE)
segments_data = cs.getSegments(scene,  oc.SEGMENT_RADIUS)
# If  no segments are found, stopt the execution of the script.
if segments_data is None:
    sys.exit()
# Unpack the segments data
[X_test, X_test_u, X_normals, X_indices, _] = segments_data
# Visualize the segments
vis = visualizer()
num_of_segments = 3
for i in np.random.uniform(low=0, high=(len(X_test_u)-1), size=num_of_segments):
    c = np.random.uniform(low=0, high=1, size=(3,))
    vis.addPointcloud(cloud=fromNumpyArray(X_test_u[int(i)]), color=c)
vis.show()
# Declare the PointNet model
object_class = object_classifier(oc.INPUT_SHAPE, oc.NUM_OF_CATEGORIES)
# Load the pretrained model
object_class.loadModel(f.OBJ_CLASS_POINTNET_LOAD)
# Run the pointnet in order to classify the segments into the different parts 
# of the object
scores = object_class.predict(X_test)
predicted_categories = object_class.getPredictedCategory(scores, min_confidence=0.2)
#cs.visualizeSegments(scene, X_test_u, predicted_categories)
object_class.visualizePrediction(seeds, scene, X_indices, predicted_categories, num_of_segments = 30)
