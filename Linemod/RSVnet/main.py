import os
import sys
from random import shuffle

import open3d as o3d

from configuration import Linemod_data, file_paths
from ground_truth_generator import groundTruthGenerator_global
from utilities import loadT

# Configuration parameters
f = file_paths
L = Linemod_data('can')

# Get the filenames of all the cloud and ground truth transformation
set_T, set_ply  = L.getTestFilesPaths()
#set_T, set_ply = L.getTrainFilePaths()
num_of_samples = len(set_T)
# Load the object pointcloud
object = o3d.io.read_point_cloud(f.OBJECT)    
# Initialize the ground truth generator
gtg = groundTruthGenerator_global(name='myRSVnetGroundTruth', object=object)
gtg.loadPointNetModel(f.SCN_CLASS_POINTNET_LOAD)
[radial_vector_length, scalar_projection] = gtg.computeRSV(object)

# Loop through all the sample clouds 
for it in range(num_of_samples):
    scene = o3d.io.read_point_cloud(set_ply[it])  
    T = loadT(set_T[it])       
    # Apply the transformation to the object point cloud in order to place 
    # it in the correct pose
    object_T = gtg.applyGroundTruthPoseToObject(T=T)    
    # In order to speed up the process, the scene is cropped to reduce the 
    # amount of points.
    cropped_scene = gtg.cropScene(object_T, scene, radius=200)
    #--> gtg.visualizeGroundTruth(object_T, cropped_scene)
    # Get the matching points, this is, those points of the scene which
    # have been classified as object points
    sample_data = gtg.getMatches(object_T, 
                                    cropped_scene, 
                                    distance_threshold=10, 
                                    radial_vector_length=radial_vector_length, 
                                    scalar_projection=scalar_projection)
    if sample_data is not None:
        #--> gtg.visualizeSampleMatchingPoints(sample_data)
        # Save the generated data every N samples
        gtg.save(f.GTG_RSVNET_G, save_every = 3000)

gtg.save(f.GTG_RSVNET_G, force_save=True)