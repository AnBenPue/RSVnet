import open3d as o3d
import copy
import numpy as np
import os
import sys

from pointcloud import *
from visualizer import visualizer
from utilities import randomShuffleeList, filterListsByIndices

# Configuration
SEED_COLOR = [1.0,0.0,0.0]
SEED_RADIUS = 4


class cloud_splitter:
    """
    Base class for the splitter. Given a point cloud, generate smaller segments.
    """
    def __init__(self, scene, segment_shape): 
        """
        Constructor of the class.

        Parameters
        ----------
        scene: o3d.geometry.PointCloud
            Pointcloud to be split.   
        """     
        self.scene = copy.deepcopy(scene)
        self.scene = computeNormals(scene, orient_normals=True)
        [self.num_of_points, self.num_of_dimensions] = segment_shape

    
    def __getNeighboringPoints_i(self, s, tree, radius, max_missing_points):
        # Get the neighbors of the seed point
        [k, idx, _] = tree.search_radius_vector_3d(s, radius)
        # Convert the indexes to numpy format
        idx = np.asarray(idx) 

        # TODO: use indices directly
        num_of_missing_points = (self.num_of_points-k) 
        missing_points_threshold = (self.num_of_points)*max_missing_points

        if k < self.num_of_points:
            if num_of_missing_points < missing_points_threshold:   
                indices = np.random.uniform(low=0, high=(k-1), size=(self.num_of_points-k))             
                idx = np.append(idx, idx[indices.astype(int)])
            else:
                return None
        return idx
    
    def __getNeighboringPoints(self, seeds, radius, max_missing_points): 
        # Get the seed points as a numpy array
        seeds = np.asarray(seeds.points)
        [num_of_seeds, _] = seeds.shape
        # Create K-d tree to search for close points in the scene pointcloud
        tree = o3d.geometry.KDTreeFlann(self.scene)
        # Loop through all the seeds and generate the segments
        s_data = zip(*[[ self.__getNeighboringPoints_i(
                                                        seeds[i], 
                                                        tree, 
                                                        radius, 
                                                        max_missing_points
                                                        ), 
                            i,
                            seeds[i]
                        ]
                        for i in range(num_of_seeds)
                      ]
                    )
        [s_neighboring_idx, seed_idx, seed_points] = s_data       

        valid_segments = np.asarray([i for i in range(len(s_neighboring_idx)) if s_neighboring_idx[i] is not None])
        s_data_valid = filterListsByIndices([s_neighboring_idx, seed_idx, seed_points], valid_segments) 

        return s_data_valid
   
    def __demeanSegment(self, seed, points):
        # In order to get the subset with a local reference frame 
        # centered in the seed point, substract the seed to all points:
        s_local = np.asarray([point - seed for point in points])
        return s_local

    def __getSegmentPointData(self, s_neighboring_idx):
        s_point_data = zip(*[
                                [
                                    np.asarray(self.scene.points)[idx], 
                                    np.asarray(self.scene.normals)[idx]
                                ]
                                for idx in s_neighboring_idx
                                ]
                            )
        return s_point_data

    def getSegments(self, seeds: o3d.geometry.PointCloud, radius: float, max_missing_points=0.5):
        # For debugging: Enable the visualization of the selected segment
        visualize = False 

        # Get 
        s_data = self.__getNeighboringPoints(seeds, radius, max_missing_points)
        [s_neighboring_idx, seed_idx, seed_points] = s_data

        if len(s_neighboring_idx) == 0:
            print('INFO: No segments found')
            return None

        s_neighboring_idx = [randomShuffleeList(idx, num_of_samples=self.num_of_points) for idx in s_neighboring_idx]

        [s_points, s_normals] = self.__getSegmentPointData(s_neighboring_idx)
        s_points_local = [self.__demeanSegment(seed_points[i], s_points[i]) for i in range(len(s_points))]

        if visualize is True:
            #TODO: fix this call, it is broken, the parameters don't match with the ones specified by the function
            self.visualizeSegments(visualize, seed_points, s_neighboring_idx)
    
        print('INFO: ' + str(len(s_points))  + ' segments found')
        new_shape = (len(s_points), self.num_of_points, self.num_of_dimensions)
        s_points_local = np.asarray(s_points_local).reshape(new_shape)
        s_points = np.asarray(s_points).reshape(new_shape)
        s_normals = np.asarray(s_normals).reshape(new_shape)
        s_neighboring_idx = np.asarray(s_neighboring_idx).reshape(len(s_points), self.num_of_points)

        return s_points_local, s_points, s_normals, np.asarray(seed_idx), copy.deepcopy(s_neighboring_idx)

    def visualizeSegment(self, visualize, seed_points, s_neighboring_idx, index):
            vis = visualizer()
            # Color the point cloud and the subset
            self.scene.paint_uniform_color([0,1,0])
            vis.addSphere(center=seed_points[index], color=[1,0,0], radius=3)
            np.asarray(self.scene.colors)[s_neighboring_idx[index], :] = [0,0,1]
            vis.addPointcloud(self.scene)                    
            vis.show()
    
    def getScene(self):
        return self.scene

    def getSeeds(self, voxel_size_scene: float, voxel_size_seeds: float,):
        """
        Downsample the pointcloud and generate the seeds.

        Parameters
        ----------
        voxel_size_scene: float
            Size of the voxel filter used to downsample the scene.
        voxel_size_seeds: float
            Size of the voxel filter used to generate the seeds.

        Returns
        ----------.
        seeds: o3d.geometry.PointCloud
            Pointcloud with the generated seeds.
        """  
        # Downsample the scene in order to reduce the amount of points and get 
        # a more uniform distribution
        self.scene = downsample(self.scene, voxel_size_scene)
        # Generate the seeds in order to split the scene point cloud. We do 
        # this by downsampling with a greater voxel size 
        seeds = downsample(self.scene, voxel_size_seeds) 
        return seeds

    def visualizeSceneAndSeeds(self, seeds):
        """
        Visualize the scene pointcloud with the generated seeds.

        Parameters
        ----------
        seeds: o3d.geometry.PointCloud
            Pointcloud containing the seeds.
        """ 
        vis = visualizer()
        vis.addPointcloud(self.scene)
        # Add a sphere in order to visualize each seed
        for s in np.asarray(seeds.points):
            vis.addSphere(center=s, color=SEED_COLOR, radius=SEED_RADIUS)
        vis.show()

    def visualizeSegments(self, seeds, segments, seeds_indices):
        # Generate a color code in order to have a different color for each
        # seed 
        seeds = np.asarray(seeds.points)
        [num_of_seeds, _] = seeds.shape
        np.random.seed(10)
        color_code = np.random.rand(num_of_seeds, 3)
        # Initialize visualizer
        vis = visualizer()
        # Add a sphere in order to visualize each seed
        for s in seeds:
            vis.addSphere(center=s, color=SEED_COLOR, radius=SEED_RADIUS)
        # Loop through all the segments, add a point cloud for each one and 
        # color it depending on the category.
        [num_of_segments, _, _] = segments.shape
        for i in range(1): #range(num_of_segments):
            segment_color = color_code[int(seeds_indices[i]), :]
            vis.addPointcloud(fromNumpyArray(segments[i]), color=segment_color)
        vis.show()
