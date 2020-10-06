import copy
import os
import sys
import time

import h5py
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from cloud_splitter import cloud_splitter
from pointcloud import computeNormals, fromNumpyArray
from rotational_subgroup_voting import rsv
from utilities import getEuclidianDistance
from visualizer import visualizer
from configuration import var_object_classification

# Configuration
v = var_object_classification

class groundTruthGenerator:
    """
    This class generates the ground truth necessary to train the SEGnet
    """
    def __init__(self, name: str, object):
        """
        Parameters
        ----------
        name: str
            name to be given when saving the dataset
        object_cloud:o3d.geometry.pointcloud
            Point cloud of the object of which the ground truth will be 
            computed
        """        
        self.name = name
        self.object = object
        # Container to save the dataset
        self.dataset_data = list()
        self.dataset_label = list()

    def applyGroundTruthPoseToObject(self, T: np.ndarray(shape=(4,4))):
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
        object_T = copy.deepcopy(self.object)
        # Translate the point cloud in order to have the cloud center in the 
        # origin
        object_T.translate(-object_T.get_center())
        # There are different cad models and not all of them have the same
        # initial orientation, in this case we had to turn it around the z axis
        T_r = R.from_euler('z', 180, degrees=True)
        object_T.rotate(T_r.as_matrix(), (0,0,0))
        # Apply the ground truth transformation      
        object_T.transform(T)
        return object_T

    def cropScene(self, object: o3d.geometry.PointCloud, 
                        scene: o3d.geometry.PointCloud, 
                        radius: float):
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
            cropped_scene = scene.select_by_index(idx)
            return cropped_scene         
    
    def getMatches(self, object, scene):
        """
        ToDo: add description

        Parameters
        ----------
        object : o3d.geometry.PointCloud
            Point cloud containing the object
        scene : o3d.geometry.PointCloud
            Point cloud containing the scene to be cropped
        """        
        # Divide the scene into small segments using the object points as seeds
        cs = cloud_splitter(scene, segment_shape=v.INPUT_SHAPE)   
        segments_data = cs.getSegments(object, v.SEGMENT_RADIUS, max_missing_points=0.4)
        # If segments are found, Save the sample data into the dataset
        if segments_data is None:
            return None            
        # Unpack the segments data
        [d_points, d_points_u, _, d_categories, d_scene_indices] = segments_data
        #--> cs.visualizeSceneAndSeeds(object)
        #--> cs.visualizeSegments(object, d_points_u, d_categories)
        self.dataset_label.append(d_categories)
        self.dataset_data.append(d_points)
        return d_points, d_points_u, d_categories, d_scene_indices

    def visualizeMatches(self, seeds, scene, segments_categories, scene_indices, num_of_points):
        vis = visualizer()
        vis.addPointcloud(scene)
        for p in np.asarray(seeds.points):
            vis.addSphere(center=p, color=[0,0,0], radius=2)

        idx = np.random.uniform(low=0, high=(len(scene_indices)-1), size=num_of_points)
        for i in idx:
            s = np.asarray(scene.points)[scene_indices[int(i)]]      
            o = np.asarray(seeds.points)[segments_categories[int(i)]]            
            vis.addSphere(center=s, color=[0,0,1], radius=2)
            vis.addSphere(center=o, color=[0,1,0], radius=3)
            vis.addLine(o, s)
        vis.show()

    def save(self, save_path: str, save_every=10, force_save=False):

        number_of_samples = self.dataset_data.__len__()
        if number_of_samples == 0:
            print('ERROR: Dataset is empty, cannot be saved.')
        elif number_of_samples >= save_every or force_save is True:
            # Get time_stamp
            ts = time.gmtime()
            time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", ts)
            # Construct file path
            data_file_name = self.name + time_stamp + '.hdf5'
            data_file_path = os.path.join(save_path,data_file_name)
            print('INFO: Saving dataset to: ' + data_file_path) 
            print('      Num of samples: ' + str(number_of_samples))
        
            # Save the new lists into an h5 file
            with h5py.File(data_file_path, 'w') as f:
                d = np.asarray(self.dataset_data)
                d = np.concatenate(d, axis=0)
                dset = f.create_dataset("data", data=d)
                d  = np.asarray(self.dataset_label)
                d = np.concatenate(d, axis=0)
                [m] = d.shape
                d = np.reshape(d,(m,1))
                dset = f.create_dataset("label", data=d, dtype='int32')
            # Empty dataset container
            self.dataset_data.clear()
            self.dataset_label.clear()
        else:
            print('INFO: Current number of samples ', number_of_samples)

    def visualizeGroundTruth(self, object, scene):
        """
        Visualize the object and the scene after applying the ground truth.
        
        Parameters
        ----------
        object : o3d.geometry.PointCloud
            Point cloud containing the object.
        scene : o3d.geometry.PointCloud
            Point cloud containing the cropped scene.         - 
        """  
        o3d.visualization.draw_geometries([object, scene]) 
