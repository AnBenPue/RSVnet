import h5py

import numpy as np

from pointcloud import getMatchingPointsIndices
from rotational_subgroup_voting import rsv
from visualizer import visualizer


def getSeedsRotationFrames(object, seeds):
    """ 
    Generate the correspondent rotation frame for each seed
    """ 
    # Get the indices of the matching points for the seeds in the object cloud
    indices = getMatchingPointsIndices(object, seeds)
    # Initialize the rotational subgroup voting module
    RSV = rsv(object)
    # Compute the rotation frames for all the points in the model
    d = RSV.computeScalarProjection()
    r = RSV.computeRadialVector(d)
    R = RSV.computeRotationFrameObject(r)
    # Select only the rotation frames for the seeds
    R_seeds = R[indices] 

    #RSV.visualizeCloudCenter()
    #q = RSV.computeRadialPoint(d)
    #RSV.visualizeRadialVector_withFrame(q, r, R, 10)

    return R_seeds

def saveSeedsRotationFrames(R_seeds, file_path):
    # Save the new lists into an h5 file
    with h5py.File(file_path, 'w') as f:
        dset = f.create_dataset("Rotation_frames", data=R_seeds)

def loadSeedsRotationFrames(file_path):
    with h5py.File(file_path, 'r') as f:
        R_seeds = f['Rotation_frames'][:]
    return R_seeds

def visualizeSeedsRotationFrames(object, seeds, R_seeds):
    vis = visualizer()
    vis.addPointcloud(seeds)
    vis.addFrame(center=[0,0,0], size=10)
    vis.addSphere(center=object.get_center(), radius=2)
    seed_points = np.asarray(seeds.points)
    
    for i in range(len(R_seeds)):
        vis.addFrame(center=seed_points[i], size=9, R=R_seeds[i])

    vis.show()