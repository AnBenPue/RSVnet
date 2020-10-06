import copy
import heapq
import os
import random
import sys
import time

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from pointcloud import computeNormals, fromNumpyArray
from utilities import (filterListsByIndices, getEuclidianDistance,
                       getKernelValue, getMinimalGeodesicDistance,
                       getPerpendicularVector, getScaledArray,
                       getUnscaledArray)
from visualizer import visualizer

# Configuration parameters
# Visualization
CANDIDATE_COLOR = [1.0,0.0,1.0]
CANDIDATE_RADIUS = 2
CLOUD_COLOR = [0.8,0.8,0.8]
CENTER_COLOR = [0.0,0.0,1.0]
CENTER_RADIUS = 2.5
RADIAL_POINT_COLOR = [1.0,0.0,0.0]
RADIAL_POINT_RADIUS = 2.5
P_PRIMA_COLOR = [0.0,1.0,0.0]
P_PRIMA_RADIUS = 2.5
VOTE_COLOR = [0.0,0.0,1.0]
VOTE_RADIUS = 0.01
FRAME_SIZE = 5

class rsv:
    """
    This class implements the procedure described in the paper:
        -   Rotational Subgroup Voting (RSV) and Pose Clustering forRobust 3D 
        Object Recognition 
        [http://openaccess.thecvf.com/content_ICCV_2017/papers/Buch_Rotational_Subgroup_Voting_ICCV_2017_paper.pdf]
    """
    def __init__(self, cloud: o3d.geometry.PointCloud, case='model', center=None):
        """
        Constructor of the class

        Parameters
        ----------
        cloud: o3d.geometry.PointCloud
            Point cloud that will be used to compute the RSV parameters
        case: str
            'model' if the values have to be computed for the model point cloud
            'scene' if the values have to be computed for the scene point cloud
        """     
        # Load the point cloud
        self.cloud = copy.deepcopy(cloud)
        # Depending on the case, the normals may have to be oriented to the 
        # camera
        if case is 'scene':
            pass
            computeNormals(self.cloud, orient_normals=True)
        elif case is 'model':
            self.cloud = computeNormals(self.cloud, orient_normals=False)
        # Save the data regarding point cloud center, points geometric 
        # coordinates and normals
        if center is None:     
            self.c = self.cloud.get_center()   
        else:
            self.c = center
        self.points = np.asarray(self.cloud.points)
        self.normals = np.asarray(self.cloud.normals)
        # Save number of points        
        [self.num_of_points, _] = np.asarray(self.points).shape
        self.points_range = range(self.num_of_points)

    def __computeScalarProjection_i(self, p, n, c):
        """
        Compute the scalar projection for a reference point. 
        (Equation 1 of the paper)

        Parameters
        ----------
        p: 1 x 3 array
            Reference point.
        n: 1 x 3 array
            Normal of the reference point.
        c: 1 x 3 array
            Center of the point cloud.

        Returns
        ----------
        d: float
            scalar projection for the reference point
        """ 
        d = (np.dot((p - c), n))
        return d

    def computeScalarProjection(self):
        """
        Compute the scalar projection for all the points in the cloud.

        Returns
        ----------
        d: number_of_points x 1 array
            scalar projection for all the points in the cloud
        """ 
        d = [self.__computeScalarProjection_i(self.points[i], 
                                              self.normals[i], 
                                              self.c) 
                                              for i in self.points_range]
        return  np.asarray(d)

    def __computeRadialVector_i(self, p, n, d, c):
        """
        Compute the radial vector for a reference point. 
        (Equation 2 of the paper)

        Parameters
        ----------
        p: 1 x 3 array
            reference point.
        n: 1 x 3 array
            normal of the reference point.
        d: float
            scalar projection of the reference point.
        c: 1 x 3 array
            Center of the point cloud.

        Returns
        ----------
        r: 1 x 3 array
            radial vector for the reference point.
        """ 
        r = c - ( p - d*n)
        return r

    def computeRadialVector(self, d):
        """
        Compute the radial vector for all the points in the cloud.

        Parameters
        ----------
        d: number_of_points x 3 array
            Scalar projection for all the points in the cloud.

        Returns
        ----------
        r: number_of_points x 3 array
            radial vector for all the points in the cloud.
        """ 
        r = [self.__computeRadialVector_i(self.points[i], 
                                               self.normals[i],
                                               d[i], 
                                               self.c) 
                                               for i in self.points_range]
        return  np.asarray(r)
    
    def computeRadialVectorLength(self, r):
        """
        Compute the radial vector length for all the points in the cloud.

        Parameters
        ----------
        r: number_of_points x 3 array
            radial vector for all the points in the cloud.

        Returns
        ----------
        l: number_of_points x 1 array
            radial vector length for all the points in the cloud.
        """ 
        l = [np.linalg.norm(v) for v in r]
        return np.asarray(l)

    def computeRadialVectorPrima_i(self, n_prima, l):
        """
        Compute the radial vector prima for a reference point. 

        Parameters
        ----------
        n_prima: 1 x 3 array
            normal of the reference point
        l: float
            radial vector length of the reference point

        Returns
        ----------
        r_prima: 1 x 3 array
            radial vector length for the reference point
        """ 
        # Get a perpendicular vector to the normal
        r_prima = getPerpendicularVector(n_prima)
        # Scale the vector to match the radial vector length (l)
        r_prima = l*r_prima
         
        return r_prima

    def computeRadialVectorPrima(self, l):
        """
        Compute the radial vector prima for all the points in the cloud

        Parameters
        ----------
        l: number_of_points x 1 array
            length for the radial vector for all the points in the cloud
            
        Returns
        ----------
        r_prima: number_of_points x 3 array
            radial vector prima for all the points in the cloud
        """ 
        r_prima = [self.computeRadialVectorPrima_i(self.normals[i],
                                                   l[i]) 
                                                   for i in self.points_range]
        return  np.asarray(r_prima)
    
    def __computeTesallation_i(self, r_prima, n_prima, increments):
        """
        Compute the tesallation of the radial vector prima for a reference 
        point

        Parameters
        ----------
        n_prima: 1 x 3 array
            normal vector prima for the reference point
        r_prima: 1 x 3 array
            radial vector prima for the reference point
        inc: tesallation_level x 1 array
            angels to rotate the radial vector prima

        Returns
        ----------
        r_prima_tesallation_i: tesallation_level x 3 array
            r_prima_tesallation for the reference point
        """ 
        r_prima_tesallation_i = [self.__rotateRadialVectorPrima(inc, 
                                                         r_prima, 
                                                         n_prima) 
                                                         for inc in increments]
        return  np.asarray(r_prima_tesallation_i)

    def computeTesallation(self, r_prima, tesallation_level):
        """
        Compute the tesallation of all the radial vector prima for all the 
        points in the cloud.

        Parameters
        ----------
        r_prima: number_of_points x 3 array
            radial vector prima for all the points in the cloud.
        tesallation_level: int
            number of votes to generate for every reference point.

        Returns
        ----------
        r_prima_tesallation: num_of_points x tesallation_level x 3 array
            r_prima_tesallation for all the points in the cloud.
        """  
        # Save the tesallation value for later use
        self.tesallation_level = tesallation_level
        # Compute the increments for the voting part
        theta = int(360 / tesallation_level) 
        increments = range(0,360,theta)
        # Compute the tesallation for all the points
        r_prima_tesallation = [self.__computeTesallation_i(r_prima[i], 
                                                    self.normals[i], 
                                                    increments) 
                                                    for i in self.points_range]
        return  np.asarray(r_prima_tesallation)

    def computeRadialPoint_i(self, p_prima, n_prima, delta):
        """
        Compute the radial point for one point

        Parameters
        ----------
        p_prima: 1 x 3 array
            reference point
        n_prima: 1 x 3 array
            normal of the reference point
        d: float
            scalar projection of the reference point

        Returns
        ----------
        q: 1 x 3 array
            radial point for the reference point
        """
        q = p_prima - np.multiply(n_prima, delta) 
        return q

    def computeRadialPoint(self, d: float):
        """
        Compute the radial point for all the points in the cloud
    
        Parameters
        ----------
        d: num_of_points x 1 array
            scalar projection

        Returns
        ----------
        q: num_of_points x 3 array
            radial vector prima for all the points in the cloud
        """ 
        q = [self.computeRadialPoint_i(self.points[i], 
                                       self.normals[i], 
                                       d[i]) 
                                       for i in self.points_range]
        return  np.asarray(q)

    def __rotateRadialVectorPrima(self, angle, r_prima, n_prima):
        """
        Rotate the radial vector prima following the Rodriguez formula

        Parameters
        ----------
        angle: float
            rotation to be applied to the vector
        r_prima: 1 x 3 array
            radial vector prima for the reference point
        n_prima: 1 x 3 array
            normal vector prima for the reference point

        Returns
        ----------
        r_prima_rot: 1 x 3 array
            rotated radial vector prima for the reference point
        """ 
        # Convert to degrees to rad
        angle = angle * np.pi / 180
        r_prima_rot = r_prima*np.cos(angle)+np.cross(n_prima,r_prima)*np.sin(angle)
        return r_prima_rot

    def __buildRotationFrame(self, r_prima, n_prima):        
        r_prima_length = np.linalg.norm(r_prima)

        col_1 = r_prima/r_prima_length
        col_3 = n_prima
        col_2 = np.cross(col_3, col_1)     

        R = [col_1, col_2, col_3]
        R_t = np.transpose(R)
      
        return R_t

    def __computeRotationFrameScene_i(self, r_prima_tesallation, n_prima):
        """
        For each point in the cloud, N votes will be generated with their 
        specific r_prima vector (with N=self.tesallation_level).
        Given r_prima and n_prima, a Rotational frame can be constructed. 
        (Equation 7 of the paper)

        Parameters
        ----------
        r_prima_tesallation: N x 1 x 3 array
            Tessellated r_prima vector 
        n_prima: 1 x 3 array
            Normal vector prima for the reference point

        Returns
        ----------
        votes: num_of_points x tesallation_level x 3 x 3 array 
            votes for the reference point
        """  
        R_r_i = [self.__buildRotationFrame(r_prima_i, n_prima) 
                                         for r_prima_i in r_prima_tesallation] 
        return  np.asarray(R_r_i)

    def computeRotationFrameObject(self, r):
                                             
        R = [self.__buildRotationFrame(r[i], self.normals[i]) for i in self.points_range] 
        return  np.asarray(R)

    def computeRotationFrameScene(self, r_prima_tesallation):
        """
        Compute the rotation frame for all the votes in the cloud

        Parameters
        ----------
        r_prima_tesallation: num_of_points x tesallation_level x 3 array 
            r_prima_tesallation for all the points in the cloud

        Returns
        ----------
        R: num_of_points x tesallation_level x 3 x 3 array 
            Rotation frames for all the votes.
        """  
        R = [self.__computeRotationFrameScene_i(r_prima_tesallation[i], 
                                           self.normals[i]) 
                                           for i in self.points_range]   
        return  np.asarray(R)

    def __computeVotes_i(self, r_prima_tesallation, q): 
        """
        Compute the votes and their rotation frames for a reference point

        Parameters
        ----------
        r_prima_tesallation: 1 x 3 array
            tessellations for the radial vector prima for the reference point
        q: 1 x 3 array
            radial point for the reference point

        Returns
        ----------
        votes: num_of_points x tesallation_level x 3 array 
            votes for the reference point
        """                
        votes = np.array([q+r_prima_i for r_prima_i in r_prima_tesallation]) 
        return votes

    def computeVotes(self, r_prima_tesallation, q):
        """
        Compute the votes for all the points in the cloud

        Parameters
        ----------
        r_prima_tesallation: 1 x 3 array
            tessellations for the radial vector prima for the reference point
        q: 1 x 3 array
            radial point for the reference point

        Returns
        ----------
        votes: num_of_points x tesallation_level x 3 array 
            votes for the reference point
        """  
        votes = [self.__computeVotes_i(r_prima_tesallation[i], q[i]) 
                                       for i in self.points_range]   
        return np.asarray(votes)
        
    def __getNeighboringPoints_i(self, s, tree, radius):
        # Get the neighbors of the seed point
        [k, idx, _] = tree.search_radius_vector_3d(s, radius)
        # Convert the indexes to numpy format
        if k > 20:
            idx = np.asarray(idx) 
        else:
            idx = None
        return idx

    def __getNeighboringPoints(self, votes_clouds, radius): 
        # Get the seed points as a numpy array
        seeds = np.asarray(votes_clouds.points)
        [num_of_seeds, _] = seeds.shape
        # Create K-d tree to search for close points in the scene pointcloud
        tree = o3d.geometry.KDTreeFlann(votes_clouds)
        # Loop through all the seeds and generate the segments
        s_data = zip(*[[ self.__getNeighboringPoints_i(
                                                        seeds[i], 
                                                        tree, 
                                                        radius
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
        [s_neighboring_idx, seed_idx, seed_points] = s_data_valid   

        num_of_neighbors = np.asarray([len(idx) for idx in s_neighboring_idx])
        max_indices = heapq.nlargest(int(0.3*len(s_neighboring_idx)), range(len(s_neighboring_idx)), num_of_neighbors.take)
        s_data_valid = filterListsByIndices([s_neighboring_idx, seed_idx, seed_points], max_indices) 
        return s_data_valid

    def __computeDensity_voteToNeighbors(self, 
                                        t_hat, 
                                        t_n,
                                        R_hat, 
                                        R_n, 
                                        sigma_t,
                                        sigma_R):
        """
        Compute the pose density for one vote given its data and the data of 
        the neighboring points. (Equation 12 of the paper)

        Parameters
        ----------
        t_hat: 1 x 3 array
            Position of the reference vote.
        t_n: N x 3 array
            Position of the neighboring votes.
        R_hat: 3 x 3 array
            Rotation frame of the reference vote.
        R_n: N x 3 x 3 array
            Rotation frame of the neighboring votes.
        sigma_t: float
            Translation bandwith.
        sigma_R: float
            Rotation bandwith.
        k: int
            Number of neighbors.
            
        Returns
        ----------
        accumulated_density: float
            Density fot the reference vote.
        """
        [k,_] = t_n.shape
        vote_density = np.asarray([self.__computeDensity_voteToVote(t_hat, t_n[i], R_hat, R_n[i], sigma_t, sigma_R) for i in range(k)])
        accumulated_density = np.sum(vote_density)
        return accumulated_density

    def __computeDensity_voteToVote(self, t_hat, t, R_hat, R, sigma_t, sigma_R):
        # Compute the distances        
        d_R = getMinimalGeodesicDistance(R_hat, R)
        if d_R <= sigma_R:
            d_t = getEuclidianDistance(t_hat, t)
            # Apply kernel
            f_K_t = getKernelValue(d_t/sigma_t)
            f_K_R = getKernelValue(d_R/sigma_R)

            density = f_K_t*f_K_R
        else:
            density = 0.0

        return density

    def computeVoteDensity(self, votes, R, sigma_t: float, sigma_R: float): 
        """
        Compute the pose density for all the points in the cloud

        Parameters
        ----------
        sigma_t: float
            Translation bandwith.
        sigma_R: float
            Rotation bandwith.
            
        Returns
        ----------
        accumulated_density: float
            Density fot the reference vote.
        """
        # Get the number of votes
        [_, tesallation_level, _] = votes.shape
        number_of_votes = tesallation_level*self.num_of_points
        # Reshape the votes and rotational frame containers
        votes_r = np.reshape(votes,(number_of_votes, 3))
        R_r = np.reshape(R,(number_of_votes, 3, 3))
        # Convert the votes into a point cloud:
        votes_cloud = fromNumpyArray(votes_r)

        print('INFO: Selecting votes neighbors')
        start = time.time()
        s_data  = self.__getNeighboringPoints(votes_cloud, sigma_t)  
        end = time.time()        
        print('\t TIME: ' + str(end - start))     
        [s_neighboring_idx, seed_idx, seed_points] = s_data 
        print('INFO: Number of valid votes: ' + str(len(seed_idx)) )
        print('INFO: Getting vote data')
        start = time.time()
        [s_t, s_R] = zip(*[[votes_r[idx], R_r[idx]] for idx in s_neighboring_idx])
        votes_r = votes_r[seed_idx]
        R_r = R_r[seed_idx]  
        end = time.time()        
        print('\t TIME: ' + str(end - start))
        print('INFO: Compute density')
        start = time.time()
        density = [self.__computeDensity_voteToNeighbors(votes_r[i], s_t[i], R_r[i], s_R[i], sigma_t, sigma_R) for i in range(len(seed_idx))]
        end = time.time() 
        print('\t TIME: ' + str(end - start))

        return density, votes_r, R_r

    def computeBestCandidates(self, votes_r, R_r, density, num_of_candidates = 0):
        # Get the number of votes
        [number_of_votes, _] = votes_r.shape
        # Reshape the votes and rotational frame containers
        #votes_r = np.reshape(votes,(number_of_votes, 3))
        #R_r = np.reshape(R,(number_of_votes, 3, 3))
        # Containers
        t_candidates = np.array([])
        R_candidates = np.array([])
        # Run until the number of candidates is satisfied
        for i in range(num_of_candidates):
            # Get the maximum density value
            density_max = np.amax(density)
            # Get the index of the element containing the max value
            indices_max = np.where(density == density_max)  
            # In case there is more than one vote with the same max value
            idx_max = indices_max[0][0]
            # Get the vote at that index          
            t_c = votes_r[idx_max]
            R_c = R_r[idx_max]
            # Set the max value to 0 in order to be able to select the next 
            # candidate
            density[idx_max] = 0
            # Save the results
            if t_candidates.size == 0:
                t_candidates = [t_c]
                R_candidates = [R_c]
            else:
                t_candidates = np.append(t_candidates, t_c)
                R_candidates = np.append(R_candidates, R_c)
        return t_candidates, R_candidates 

    def visualizeDensity(self, votes_r, density, t_candidates, R_candidates, model):
        """
        Visualize the vote density 

        Parameters
        ----------
        votes 

        R

        density

        model:o3d.geometry.PointCloud

        num_of_candidates: int
            Number of candidates to be shown.
        """
        # Initialize visualizer
        vis = visualizer()
        vis.addPointcloud(self.cloud, CLOUD_COLOR) 
        # Convert the votes into a point cloud
        votes_cloud = fromNumpyArray(votes_r)
        # Scale the density values to be within [0,1] 
        color_code, _, _ = getScaledArray(density, high=1, low=0)        
        # Color Coding the votes: First, Paint the cloud uniformly in order for 
        # the points to have a color attribute
        votes_cloud.paint_uniform_color([1, 1, 1])
        # Color Coding the votes: Second, Assign a color to each point
        # depending on its density value
        votes_color_coded = np.array([[c, 0, 0] for c in color_code])
        votes_cloud.colors = o3d.utility.Vector3dVector(votes_color_coded)
        vis.addPointcloud(votes_cloud)
        # Run until the number of candidates is satisfied
        for i in range(len(t_candidates)):           
            t_c = t_candidates[i]
            R_c = R_candidates[i]
            # Sphere to visualize the candidate
            vis.addSphere(t_c, CANDIDATE_COLOR, CANDIDATE_RADIUS)
            vis.addFrame(t_c, FRAME_SIZE, R=R_c)
            # Candidate object point cloud
            vis.addPointcloud(copy.deepcopy(model), t=t_c, R=R_c)
        # Show results
        vis.show()      

    def visualizeCloud(self):
        """
        Visualize the input point cloud
        """ 
        o3d.visualization.draw_geometries([self.cloud])
    
    def visualizeVotes(self, votes):
        """
        Visualize the input point cloud with the generated votes
        """ 
        # Initialize visualizer
        vis = visualizer()
        vis.addPointcloud(self.cloud, color=CLOUD_COLOR) 
        # Get the number of votes
        [_, tesallation_level, _] = votes.shape
        number_of_votes = tesallation_level*self.num_of_points
        # Reshape the votes container
        votes_r = np.reshape(votes,(number_of_votes, 3))          
        # Convert the votes into a point cloud        
        votes_cloud = fromNumpyArray(votes_r)
        vis.addPointcloud(votes_cloud, color=VOTE_COLOR) 
        # Show results
        vis.show()

    def visualizeCloudCenter(self):
        """
        Visualize the input point cloud with its center
        """ 
        # Initialize visualizer
        vis = visualizer()
        vis.addPointcloud(self.cloud, CLOUD_COLOR) 
        vis.addSphere(self.c, CENTER_COLOR, CENTER_RADIUS)
        # Show results
        vis.show()

    def visualizeVote(self, votes, R, q, num_of_votes:int):       
        # Initialize visualizer
        vis = visualizer()
        vis.addPointcloud(self.cloud, CLOUD_COLOR) 
        vis.addSphere(self.c, CENTER_COLOR, CENTER_RADIUS)
        vis.addFrame(self.c, FRAME_SIZE*2)
        vis.addFrame([0.0,0.0,0.0], FRAME_SIZE*2)
        # Add visualization aid to as many votes as requested
        for i in range(num_of_votes):
            # Select a random vote
            index  = random.randint(0, self.num_of_points)
            # Sphere to visualize p_prima
            vis.addSphere(self.points[index], P_PRIMA_COLOR, P_PRIMA_RADIUS)
            # Line from p_prima to the radial point
            vis.addLine(self.points[index], q[index])
            # Sphere to visualize the radial point
            vis.addSphere(q[index], RADIAL_POINT_COLOR, RADIAL_POINT_RADIUS)
            # Get the vote data
            t_i = votes[index]
            R_i = R[index]
            # Get the tesallation_level 
            [_, tesallation_level, _] = votes.shape
            # Loop through all the votes generated for the reference point 
            # p_prima
            for j in range(tesallation_level):             
                # Sphere to visualize the vote
                vis.addSphere(t_i[j], VOTE_COLOR, VOTE_RADIUS)
                # Reference frame for the vote
                vis.addFrame(t_i[j], FRAME_SIZE, t=None, R=R_i[j])
                # Line from the vote to the radial point
                vis.addLine(t_i[j], q[index])
        # Show results
        vis.show()

    def visualizeRadialVector(self, q, r_prima, num_of_samples):
        # Initialize visualizer
        vis = visualizer()
        vis.addPointcloud(cloud=self.cloud, color=CLOUD_COLOR) 
        vis.addSphere(self.c, CENTER_COLOR, CENTER_RADIUS)
        # Randomly select a subsample of points
        indices = range(self.num_of_points)
        for index in np.random.choice(indices, num_of_samples):
            # Sphere to visualize the reference point
            vis.addSphere(self.points[index], P_PRIMA_COLOR, P_PRIMA_RADIUS)
            # Sphere to visualize the radial point
            vis.addSphere(q[index], RADIAL_POINT_COLOR, RADIAL_POINT_RADIUS)
            # Sphere to visualize the vote generated by r_prima
            vote = q[index]+r_prima[index]
            vis.addSphere(vote, VOTE_COLOR, CENTER_RADIUS)
            # Line from the vote to the radial point
            vis.addLine(vote, q[index])
            # Line from the p_prima to the radial point
            vis.addLine(self.points[index], q[index])
        # Show results
        vis.show()

    def visualizeRadialVector_withFrame(self, q, r_prima, R, num_of_samples):
        # Initialize visualizer
        vis = visualizer()
        vis.addPointcloud(cloud=self.cloud, color=CLOUD_COLOR) 
        vis.addSphere(self.c, CENTER_COLOR, CENTER_RADIUS)
        # Randomly select a subsample of points
        indices = range(self.num_of_points)
        for index in np.random.choice(indices, num_of_samples):
            # Sphere to visualize the reference point
            vis.addSphere(self.points[index], P_PRIMA_COLOR, P_PRIMA_RADIUS)
            # Sphere to visualize the radial point
            vis.addSphere(q[index], RADIAL_POINT_COLOR, RADIAL_POINT_RADIUS)
            # Sphere to visualize the vote generated by r_prima
            vote = q[index]+r_prima[index]
            vis.addSphere(vote, VOTE_COLOR, VOTE_RADIUS)
            # Line from the vote to the radial point
            vis.addLine(vote, q[index])
            # Line from the p_prima to the radial point
            vis.addLine(self.points[index], q[index])
            # Add reference frame
            vis.addFrame(center=self.points[index], size=FRAME_SIZE, R=R[index])
        # Show results
        vis.show()

    def getRelativeRotationFrame_i(self, R_o, R_s_set):
        R = list()
        R_o = np.asmatrix(R_o)
        for R_s in R_s_set:        
            R_s = np.asmatrix(R_s)
            R_o_t = np.transpose(R_o)      
            R_r = np.matmul(R_s, R_o_t)
            R.append(R_r)
        return R

    def getRelativeRotationFrame(self, R_o, R_s_all):
        R_r = [self.getRelativeRotationFrame_i(R_o[i], R_s_all[i]) for i in range(len(R_s_all))]
        return R_r
