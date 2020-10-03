import copy
import os
import sys

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from configuration import var_RSV, var_object_classification, file_paths
from pointcloud import *
from rotational_subgroup_voting import rsv
from visualizer import visualizer

# Configuration parameters
v1 = var_RSV
v2 = var_object_classification
f = file_paths

leaf_size = 10
# Load object point cloud
object = o3d.io.read_point_cloud(f.OBJECT)
# Center the cloud in the origin
object = downsample(object, leaf_size)
object.translate(-object.get_center())

# Load the scene point cloud
scene = o3d.io.read_point_cloud(f.OBJECT)
scene = downsample(scene, leaf_size)
scene.paint_uniform_color([0,0.5,1])
o3d.visualization.draw_geometries_with_editing([scene])
#scene = downsample(scene, leaf_size)
"""
scene.translate([500,100,100])
T_r = Rotation.from_euler('z', 65, degrees=True)
scene.rotate(T_r.as_matrix())
T_r = Rotation.from_euler('x', 63, degrees=True)
scene.rotate(T_r.as_matrix())
T_r = Rotation.from_euler('y', 12, degrees=True)
scene.rotate(T_r.as_matrix())
"""
#--> o3d.visualization.draw_geometries([object, scene])

# Initialize RSV for e the object
RSV_o = rsv(object)
#--> RSV_o.visualizeCloudCenter()
# Compute the rotation frames for all the points in the object
d_o = RSV_o.computeScalarProjection()
r_o = RSV_o.computeRadialVector(d_o)
R_o = RSV_o.computeRotationFrameObject(r_o)

# Initialize RSV for e the scene
RSV_s = rsv(scene)
#--> RSV_s.visualizeCloudCenter()
# Compute the votes
d_s = RSV_s.computeScalarProjection()
r_s = RSV_s.computeRadialVector(d_s)
l_s = RSV_s.computeRadialVectorLength(r_s)
q_s = RSV_s.computeRadialPoint(d_s)
r_prima_s = RSV_s.computeRadialVectorPrima(l_s)
r_prima_tesallation_s = RSV_s.computeTesallation(r_prima_s, v1.TESALLATION_LEVEL)
R_s = RSV_s.computeRotationFrameScene(r_prima_tesallation_s)
votes_s = RSV_s.computeVotes(r_prima_tesallation_s, q_s)
RSV_s.visualizeVote(votes_s, R_s, q_s, num_of_votes=1)
# Get the relative rotation frame
R_r = RSV_s.getRelativeRotationFrame(R_o, R_s)
#--> RSV_s.visualizeVote(votes_s, R_r, q_s, num_of_votes=5)

candidate_cloud = copy.deepcopy(object)
candidate_cloud.paint_uniform_color([0,0,1])
density_s = RSV_s.computeVoteDensity(votes_s, R_r, v1.SIGMA_t, v1.SIGMA_R)
RSV_s.visualizeDensity(votes_s, R_r, density_s, candidate_cloud, 1)

