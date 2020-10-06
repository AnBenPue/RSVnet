import copy

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from utilities import getEuclidianDistance


"""
This file contains functions used through the project when working with
pointclouds
"""

def downsample(cloud: o3d.geometry.PointCloud, v_size: float, with_trace=False):
    """
    Downsample the point cloud with a voxel grid with the defined voxel size

    Parameters
    ----------
    cloud: o3d.geometry.PointCloud
        Cloud to be downsampled.
    v_size: float
        Voxel size to downsample into.
    with_trace: bool
        Select recording of point cloud index before downsampling.

    Returns
    ----------
    cloud: o3d.geometry.PointCloud
        Downsampled point cloud
    """  
    if with_trace is True:
        [cloud, indices] = cloud.voxel_down_sample_and_trace(v_size, min_bound=[0,0,0], max_bound=[1000,1000,1000])
        return [cloud, indices]
    else:
        cloud = cloud.voxel_down_sample(v_size)
        return cloud
    
def computeNormals(cloud: o3d.geometry.PointCloud, orient_normals=False):
    """
    Compute the normals for the points in the point cloud.

    Parameters
    ----------
    cloud: o3d.geometry.PointCloud
        Cloud to compute the normals.
    orient_normals: bool
        Select if the normals have to be oriented towards the camera.

    Returns
    ----------
    cloud: o3d.geometry.PointCloud
        Cloud with the normals computed
    """  
    cloud.estimate_normals()
    if orient_normals is True: 
        cloud.orient_normals_towards_camera_location()      
    return cloud

def fromNumpyArray(points, normals=None):
    """
    Construct an open3d pointcloud from a numpy array.

    Parameters
    ----------
    points: NUM_OF_POINTS x 3 array
        Points data for the cloud.
    normals: NUM_OF_POINTS x 3 array
        Normals data for the cloud.

    Returns
    ----------
    cloud: o3d.geometry.PointCloud
        Open3d point cloud.
    """  
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points) 
    if normals is not None:
        cloud.normals =  o3d.utility.Vector3dVector(normals)
    return cloud

def getMatchingPointsIndices(cloud_A, cloud_B):
    """
    Given two point clouds (A and B), find the closest points in B for all the
    points of A.

    Parameters
    ----------
    cloud_A: o3d.geometry.PointCloud
        Cloud with the points which we want to find the closest neighbors.
    cloud_B: o3d.geometry.PointCloud
        Cloud to be used as reference.

    Returns
    ----------
    indices
        Indices for the matching points (regarding cloud_A)
    """
    # Build the Kd-tree to search through the points of cloud_A
    cloud_A_tree = o3d.geometry.KDTreeFlann(cloud_A)
    indices = np.array([], dtype=int)
    # Loop through all the points of the cloud_B
    for p in cloud_B.points:
        [k, idx, _] = cloud_A_tree.search_knn_vector_3d(p, 1)
        indices = np.append(indices, int(idx[0]))
    return indices

def getClosestPointsIndices(cloud_A, cloud_B, radius):
    # Build the Kd-tree to search through the points of cloud_A
    cloud_A_tree = o3d.geometry.KDTreeFlann(cloud_A)
    indices = np.array([], dtype=int)
    # Loop through all the points of the cloud_B
    for p in cloud_B.points:
        [k, idx, _] = cloud_A_tree.search_radius_vector_3d(p, radius)
        indices = np.append(indices, idx)
    return indices
    

def applyGroundTruthPoseToObject(object, T: np.ndarray(shape=(4,4))):
    """
    Apply the geometrical transformation given by the Linemod ground truth 
    in order to have the cad model of the object with the correct pose.

    Parameters
    ----------
    T: 4 x 4 array
        Transformation matrix

    Returns
    -------
    object_T : o3d.geometry.PointCloud
        Transformed object point cloud.
    """
    object_T = copy.deepcopy(object)
    # Convert units from meters to mm
    T[0,3] = T[0,3] * 1000
    T[1,3] = T[1,3] * 1000
    T[2,3] = T[2,3] * 1000
    # Translate the point cloud in order to have the cloud center in the 
    # origin
    object_T.translate(-object_T.get_center())
    # There are different cad models and not all of them have the same
    # initial orientation, in this case we had to turn it around the z axis
    T_r = R.from_euler('z', 180, degrees=True)
    object_T.rotate(T_r.as_matrix())
    # Apply the ground truth transformation      
    object_T.transform(T)
    return object_T

def cropScene(object, scene, radius: float): 
    """
    Select the points of the scene point cloud which are within a certain 
    distance from the object center.
        After applying the ground truth pose to the object, the center of 
        the object 'c' is taken as a reference point. The resulting cloud 
        only keeps those points which are withtin a sphere centered in 'c' 
        with radius 'radius'.

    Parameters
    ----------
    object : o3d.geometry.PointCloud
        Point cloud containing the object
    scene : o3d.geometry.PointCloud
        Point cloud containing the scene to be cropped
    radius : float
        Distance threshold
    
    Returns
    -------
    cropped_scene : o3d.geometry.PointCloud
        Cropped scene point cloud
    """
    # Build the Kd-tree to search through the points of the scene pointcloud
    scene_tree = o3d.geometry.KDTreeFlann(scene)
    # Get the center of the object
    c = object.get_center()
    # Get the neighboring points (points of the scene which are closer than
    # the radius to the object center)
    [k, idx, _] = scene_tree.search_radius_vector_3d(c, radius)
    if k == 0:
        print('ERROR: cropping failed.')
    else:
        # Crop scene point cloud by selecting only the neighboring points 
        cropped_scene = scene.select_down_sample(idx)
        return cropped_scene 

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def applyPoseToObject(object, T: np.ndarray(shape=(4,4))):
    object_T = copy.deepcopy(object)
    object_T.translate(-object_T.get_center())
    object_T.transform(T)
    return object_T

def applyGroundTruthPoseToObject(object, T: np.ndarray(shape=(4,4))):
    """
    Apply the geometrical transformation given by the Linemod ground truth 
    in order to have the cad model of the object with the correct pose.

    Parameters
    ----------
    T: 4 x 4 array
        Transformation matrix

    Returns
    -------
    object_T : o3d.geometry.PointCloud
        Transformed object point cloud.
    """
    object_T = copy.deepcopy(object)
    # Translate the point cloud in order to have the cloud center in the 
    # origin
    object_T.translate(-object_T.get_center())
    # There are different cad models and not all of them have the same
    # initial orientation, in this case we had to turn it around the z axis
    T_r = R.from_euler('z', 180, degrees=True)
    object_T.rotate(T_r.as_matrix())
    # Apply the ground truth transformation      
    object_T.transform(T)
    return object_T

def getFurthestPoint(cloud, reference):
    "Find the furthes point in a cloud regarding a reference point"
    points = np.asarray(cloud.points)
    distances_referenceTp = np.asarray([getEuclidianDistance(reference, p) for p in points]) 
    d_max_ctp = np.max(distances_referenceTp)
    idx_max = np.where(distances_referenceTp == d_max_ctp)
    furthest_point = points[idx_max]
    return furthest_point[0]

class icp():
    def __init__(self, source, target, trans_init, threshold):
        self.source = source
        self.target = target
        self.trans_init = trans_init
        self.threshold = threshold

        print("Initial alignment")
        evaluation = o3d.registration.evaluate_registration(source, target,
                                                        threshold, trans_init)
        print(evaluation)

    def point_to_point(self, vis=False):
        print("Apply point-to-point ICP")
        reg_p2p = o3d.registration.registration_icp(
            self.source, self.target, self.threshold, self.trans_init,
            o3d.registration.TransformationEstimationPointToPoint(),
            o3d.registration.ICPConvergenceCriteria(relative_fitness = 1e-06, 
                                                    relative_rmse = 1e-06, 
                                                    max_iteration = 300))
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        print("")

        if vis is True:
             draw_registration_result(self.source, self.target, reg_p2p.transformation)
        return reg_p2p.transformation

    def point_to_plane(self, vis=False):    
        print("Apply point-to-plane ICP")
        reg_p2l = o3d.registration.registration_icp(
            self.source, self.target, self.threshold, self.trans_init,
            o3d.registration.TransformationEstimationPointToPlane(),
            o3d.registration.ICPConvergenceCriteria(relative_fitness = 1e-06, 
                                                    relative_rmse = 1e-06, 
                                                    max_iteration = 300))
        print(reg_p2l)
        print("Transformation is:")
        print(reg_p2l.transformation)
        print("")

        if vis is True:
            draw_registration_result(self.source, self.target, reg_p2l.transformation)
        return reg_p2l.transformation

