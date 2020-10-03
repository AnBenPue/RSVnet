import open3d as o3d

from configuration import Linemod_data, file_paths, var_object_classification
from ground_truth_generator import groundTruthGenerator
from pointcloud import downsample, fromNumpyArray
from seeds_rotation_frames import (getSeedsRotationFrames,
                                   saveSeedsRotationFrames)
from utilities import loadT

# Configuration parameters
v = var_object_classification
f = file_paths
L = Linemod_data('can')

# Get the filenames of all the cloud and ground truth transformation
#set_T, set_ply  = L.getTestFilesPaths()
set_T, set_ply = L.getTrainFilePaths()
num_of_samples = len(set_T)
# Load the object pointcloud
object = o3d.io.read_point_cloud(f.OBJECT)
# Downsample the point cloud with a voxel size big enough to reduce to 100 the
# number of points in the object model. This points will be used as seeds.
seeds = downsample(object, v.OBJECT_VOXEL_SIZE)
o3d.io.write_point_cloud(f.OBJECT_SEEDS, seeds)
# Initialize the ground truth generator
gtg = groundTruthGenerator(name='myRSVnetGroundTruth', object=seeds)
# Generate the correspondent rotation frame for each seed
R_seeds = getSeedsRotationFrames(object, seeds)
saveSeedsRotationFrames(R_seeds, f.GTG_OBJECT_SEEDS_R)

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
    #-->  sgtg.visualizeGroundTruth(object_T, cropped_scene)
    # Get the matching points, this is, those points of the scene which
    # have been classified as object points
    [d_points, d_points_uncentered, d_categories, d_scene_indices] = gtg.getMates(object_T, cropped_scene)
    #--> gtg.visualizeMatches(object_T, cropped_scene, d_categories, d_scene_indices, num_of_points=3)
    #--> gtg.visualizeMatches(object_T, cropped_scene.translate([100,100,100]), d_categories, d_scene_indices, num_of_points=3)
    # Save the generated data every N samples
    gtg.save(f.GTG_OBJECT_SEGMENTATION_DATA, save_every = 1000)

gtg.save(f.GTG_OBJECT_SEGMENTATION_DATA, force_save=True)

