import json
import os
from json.decoder import JSONDecodeError

import numpy as np
import open3d as o3d

from pointcloud import (applyGroundTruthPoseToObject, applyPoseToObject,
                        cropScene, getFurthestPoint)
from utilities import getAngleBetweenVectors, getEuclidianDistance
from visualizer import visualizer


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class evaluation_metrics():
    def __init__(self, file_path):
        self.loadMetrics(file_path)

    def loadMetrics(self, file_path):
        if os.path.isfile(file_path):
            try:
                with open(file_path, "r") as read_file:
                    self.metrics = json.load(read_file)
            except JSONDecodeError:
                print('ERROR: json file cannot be parsed')
                self.metrics = {}
        else:
            x =  '{}'
            self.metrics = json.loads(x)
        return self.metrics

    def saveMetrics(self, file_path):
        with open(file_path, "w") as write_file:
            json.dump(self.metrics, write_file,cls=NumpyEncoder, indent=4)
    
    def printMetrics(self):
        print(self.metrics)

    def addData(self, sample, key, value):
        #print('INFO: Saving to item: ' + sample + ' :{ ' + key + ' : ' + str(value) + ' }')
        try:
            item = self.metrics[sample]
            item.update({key:value})
        except KeyError:
            new_item = {sample:{key:value}}
            self.metrics.update(new_item)    

    def getData(self, sample, key): 
        try:
            sample_data = self.metrics[sample]
            return sample_data[key]
        except KeyError:
            return None
        
    def computeDiagonal(self, object, vis=False):
        points = np.asarray(object.points)

        c = (object.get_center())
        p1 = getFurthestPoint(object, c)  
        p2 = getFurthestPoint(object, p1) 
        
        diagonal = getEuclidianDistance(p1, p2)

        if vis is True:
            vis = visualizer()
            vis.addPointcloud(object)
            vis.addSphere(center=p1, color=[0,1,0], radius=2)
            vis.addSphere(center=p2, color=[0,0,1], radius=2)
            vis.addLine(p1,p2)
            vis.show()

        return diagonal

    def computeCD(self, object_gt, object_p):
        c_gt = (object_gt.get_center())
        c_p = (object_p.get_center())

        d = getEuclidianDistance(c_gt, c_p)

        if d < 10:
            sample_valid = 'X'
        else:
            sample_valid = '-'
        
        return d, sample_valid

    def computeADD(self, object_gt, object_p, diagonal):
        points_gt = np.asarray(object_gt.points)
        points_p = np.asarray(object_p.points)

        num_of_points = len(points_gt)
        distances_ptp = np.asarray([getEuclidianDistance(points_gt[it], points_p[it]) for it in range(num_of_points)])
        average_distance = np.average(distances_ptp)      

        if average_distance < 0.1*diagonal:
            sample_valid = 'X'
        else:
            sample_valid = '-'

        return average_distance, sample_valid

    def computeRadialDistance(self, T_gt, T_p, axis, Threshold):

        if axis == 'x':
            v1 = T_gt[0:3,0]
            v2 = T_p[0:3,0]
        elif axis == 'y':
            v1 = T_gt[0:3,1]
            v2 = T_p[0:3,1]
        elif axis == 'z':
            v1 = T_gt[0:3,2]
            v2 = T_p[0:3,2]

        angle = getAngleBetweenVectors(v1, v2)

        if angle < Threshold:
            sample_valid = 'X'
        else:
            sample_valid = '-'

        return angle, sample_valid

    def isSampleValid(self, sample:str):
        try:
            self.metrics[sample]
            return True
        except KeyError:
            return False

    def evaluateOneSample(self, sample, scene_file_path, object, method='ADD', diagonal=0.0, visualize=False):
        # Load the corresponding scene pointcloud
        scene = o3d.io.read_point_cloud(scene_file_path)    
        # Load the sample transformations
        T_gt = self.getData(sample, 'T_ground_truth')
        T = self.getData(sample, 'T')
        T_icp_ptpoint = self.getData(sample, 'T_icp_ptpoint')
        T_icp_ptplane = self.getData(sample, 'T_icp_ptplane')

        if T_gt is None or T_icp_ptpoint is None or T_icp_ptplane is None or T is None:
            return 'None'
    
        T_gt = np.asarray(T_gt)
        T = np.asarray(T)
        T_icp_ptplane = np.asarray(T_icp_ptplane)
        T_icp_ptpoint = np.asarray(T_icp_ptpoint)

        # Apply transformations to the object
        object_gt = applyGroundTruthPoseToObject(object, T_gt)
        object_T = applyPoseToObject(object, T)
        object_icp_ptplane = applyPoseToObject(object, T_icp_ptplane)
        object_icp_ptpoint = applyPoseToObject(object, T_icp_ptpoint)

        # Compute evaluation metric
        if method == 'ADD':
            m_value_T, m_valid_T = self.computeADD(object_gt, object_T, diagonal)
            m_value_icp_ptplane, m_valid_icp_ptplane = self.computeADD(object_gt, object_icp_ptplane, diagonal)
            m_value_icp_ptpoint, m_valid_icp_ptpoint = self.computeADD(object_gt, object_icp_ptpoint, diagonal)

        elif method == 'CD':
            m_value_T, m_valid_T = self.computeCD(object_gt, object_T)
            m_value_icp_ptplane, m_valid_icp_ptplane = self.computeCD(object_gt, object_icp_ptplane)
            m_value_icp_ptpoint, m_valid_icp_ptpoint = self.computeCD(object_gt, object_icp_ptpoint)

        elif method == 'RD':
            max_angle = 5
            m_value_T, m_valid_T = self.computeRadialDistance(T_gt, T, 'z', Threshold=max_angle)
            m_value_icp_ptplane, m_valid_icp_ptplane = self.computeRadialDistance(T_gt, T_icp_ptplane, 'z', Threshold=max_angle)
            m_value_icp_ptpoint, m_valid_icp_ptpoint = self.computeRadialDistance(T_gt, T_icp_ptpoint, 'z', Threshold=max_angle)
    

        msg = 'Sample data: ' + sample + ' Result: | ' + str(m_valid_T) + ' , ' + str(m_valid_icp_ptplane) + ' , ' + str(m_valid_icp_ptpoint) + '  | '
        print(msg)

        if visualize is True:  
            vis = visualizer()

            #vis.addPointcloud(object_T, color=[1, 0.706, 0])
            vis.addPointcloud(object_icp_ptplane, color=[0, 1, 0])
            #vis.addPointcloud(object_icp_ptpoint, color=[0, 0.706, 0])
            
            scene = cropScene(object_gt, scene, radius = 200)
            vis.addPointcloud(scene)
            vis.show()

        return [m_valid_T, m_valid_icp_ptplane, m_valid_icp_ptpoint]
