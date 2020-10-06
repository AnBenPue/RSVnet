import copy
import os
import sys
import time

import h5py
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from cloud_splitter import cloud_splitter
from configuration import var_scene_classification
from pointcloud import computeNormals, downsample, fromNumpyArray
from PointNet_model import pointnet_model
from rotational_subgroup_voting import rsv
from utilities import getEuclidianDistance
from visualizer import visualizer

# Configuration parameters
v = var_scene_classification


class groundTruthGenerator_global:

    def __init__(self, name: str, object: o3d.geometry.PointCloud):
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
        self.dataset = np.array([])
        self.dataset_out = np.array([])
        self.dataset_seed = np.array([])
    
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

    def computeRSV(self, object:o3d.geometry.PointCloud):
        """
        Compute the rotational subgroup voting data for the object point cloud.
        
        Parameters
        ----------
        object : o3d.geometry.PointCloud
            Point cloud containing the object

        Returns
        -------
        radial_vector_length: NUM_OF_POINTS x 1 array
            Parameter r for all the points in the object.
        scalar_projection: NUM_OF_POINTS x 1 array
            Parameter delta for all the points in the object.
        """
        RSV = rsv(object, 'model')
        scalar_projection = RSV.computeScalarProjection()
        r = RSV.computeRadialVector(d=scalar_projection) 
        radial_vector_length = RSV.computeRadialVectorLength(r=r)
        return radial_vector_length, scalar_projection

    def getMatches(self, object:o3d.geometry.PointCloud, 
                         scene:o3d.geometry.PointCloud, 
                         distance_threshold:float, 
                         radial_vector_length, 
                         scalar_projection):
        visualize = False
        # Container to save the sample data
        sample_data  = np.array([])
        # Construct kd-tree in order to search through the object points
        object_tree = o3d.geometry.KDTreeFlann(object)
        # Compute normals of the scene point cloud 
        scene = computeNormals(cloud=scene, orient_normals=True)
        scene_points = np.asarray(scene.points)
        scene_normals = np.asarray(scene.normals)
        [number_of_points_scene,_]=scene_points.shape     
  
        scene_points_indices = np.array([], dtype=int)
        # Loop through all the points in the scene point cloud
        for i in range(number_of_points_scene): 
            # Find the closest point of the object to the scene point
            [k, idx, _] = object_tree.search_knn_vector_3d(scene_points[i], 1)
            object_point = np.asarray(object.points)[idx] 
            # Compute the euclidian distance between the object and scene point
            dist = getEuclidianDistance(scene_points[i],object_point)
            # If the points are close enough, save the data
            if dist < distance_threshold: 
                scene_points_indices = np.append(scene_points_indices, i)
                # Storing the sample data:
                # [scene_point(x,y,z),
                # radial_vector_length,
                # scalar_projection,
                # normal]    
                point_data = np.hstack([scene_points[i], 
                                        radial_vector_length[idx], 
                                        scalar_projection[idx], 
                                        scene_normals[i]])   
                if sample_data.size == 0:
                    sample_data = point_data
                else:
                    sample_data = np.row_stack([sample_data, point_data])

        if visualize is True:
            vis = visualizer()
            scene.paint_uniform_color([0,1,0])
            np.asarray(scene.colors)[scene_points_indices[1:], :] = [0,0,1]
            vis.addPointcloud(scene)
            vis.show()

        scene_tree = o3d.geometry.KDTreeFlann(scene)
        indices = np.random.uniform(low=0, high=(sample_data.__len__()-1), size=50)
        for it in indices:
            it = int(it)
            seed = sample_data[it, 0:3]
            [k, idx, _] = scene_tree.search_radius_vector_3d(seed, v.SEGMENT_RADIUS)
            
            if visualize is True:
                vis = visualizer()
                scene.paint_uniform_color([0,1,0])
                np.asarray(scene.colors)[idx[1:], :] = [0,0,1]
                vis.addPointcloud(scene)
                vis.addSphere(center=seed, color=[1,0,0], radius=2)
                vis.show()

            idx = np.asarray(idx)
            np.random.shuffle(idx)
            idx = idx[0:v.NUM_OF_POINTS]
            points = copy.deepcopy(np.asarray(scene.points)[idx])
            points = np.asarray([[p - seed for p in points]])
            normals = [copy.deepcopy(np.asarray(scene.normals)[idx])]

            if visualize is True:
                vis = visualizer()
                scene.paint_uniform_color([0,1,0])
                np.asarray(scene.colors)[idx[1:], :] = [0,0,1]
                vis.addPointcloud(scene)
                vis.addSphere(center=seed, color=[1,0,0], radius=2)
                vis.show()
            
            [_, m, _] = points.shape
            if m == v.NUM_OF_POINTS:
                gf_vector = self.getGlobalFeatureVectors([points])
            else:
                return None
        
            if self.dataset.size == 0:
                self.dataset = np.asarray([sample_data[it, :]])
                self.dataset_out = np.asarray(gf_vector)
                self.dataset_cloud = np.asarray(points)
                self.dataset_normals = np.asarray(normals)
            else:
                self.dataset = np.row_stack([self.dataset, [sample_data[it, :]]])
                self.dataset_out = np.row_stack([self.dataset_out, gf_vector])
                self.dataset_cloud = np.row_stack([self.dataset_cloud, points])
                self.dataset_normals = np.row_stack([self.dataset_normals, normals])
                
        return sample_data       

    def save(self, save_path: str, save_every=10, force_save=False):
        """
        Save the dataset into an .hdf5 file.

        Parameters
        -------
        save_path: str
            Path to save the file. As a name, the file will be given the same 
            assigned to the ground truth generator when constructed
        
        save_every: int
            Save the dataset once this number of samples has been reached
        """
        number_of_samples = self.dataset.__len__()
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
                d = self.dataset
                dset = f.create_dataset("Cloud", data=self.dataset_cloud)
                dset = f.create_dataset("Normals", data=self.dataset_normals)
                dset = f.create_dataset("Seed", data=d[:, 0:3])
                dset = f.create_dataset("RadialVectorLength", data=d[:, 3])
                dset = f.create_dataset("Projection", data=d[:, 4])
                dset = f.create_dataset("Seed_Normal", data=d[:, 5:8])
                dset = f.create_dataset("GlobalFeatureVectors", data=self.dataset_out)
            # Empty dataset container
            self.dataset = np.array([])
        else:
            print('INFO: Current number of samples ',number_of_samples)

    def loadPointNetModel(self, path_to_model: str):
        """
        Load and initialize a trained pointnet model           - 

        Parameters
        ----------
        path_to_model : str
            Path to the trained pointnet model.
        """    
        self.pointnet = pointnet_model(v.INPUT_SHAPE, v.NUM_OF_CATEGORIES)  
        self.pointnet.loadModel(path_to_model)
        
    def getLocalFeatureVectors(self, X_test):
        """
        Compute the local feature vectors for all the points in a pointcloud           - 

        Parameters
        ----------
        X_test : NUM_OF_POINTS x NUM_OF_DIMENSIONS  array
            Data to be tested
    
        Returns
        -------

        """    
        if not hasattr(self, 'pointnet'):
            print('ERROR: Pointnet model not found. please run loadPointNetModel().')
        else:
            lf_vectors = self.pointnet.predictLocalFeatureVectors(X_test)
            new_shape = (v.NUM_OF_POINTS, v.NUM_OF_FEATURES) 
            lf_vectors = np.reshape(lf_vectors, new_shape)
            return lf_vectors

    def getGlobalFeatureVectors(self, X_test):
        """
        Compute the local feature vectors for all the points in a pointcloud           - 

        Parameters
        ----------
        X_test : NUM_OF_POINTS x NUM_OF_DIMENSIONS  array
            Data to be tested
        
        Returns
        -------
        """    
        if not hasattr(self, 'pointnet'):
            print('ERROR: Pointnet model not found. please run loadPointNetModel().')
        else:
            gf_vectors = self.pointnet.predictGlobalFeatureVectors(X_test)
            return gf_vectors
