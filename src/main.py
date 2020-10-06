import sys
import time

import numpy as np
import open3d as o3d

from cloud_splitter import cloud_splitter
from configuration import (Linemod_data, file_paths, var_object_classification,
                           var_RSV, var_scene_classification)
from pointcloud import (applyGroundTruthPoseToObject, cropScene, downsample,
                        fromNumpyArray, icp)
from PointNet_model import object_classifier, scene_classifier
from rotational_subgroup_voting import rsv
from RSVnet import RSVnet_model
from seeds_rotation_frames import loadSeedsRotationFrames
from utilities import buildT, loadT
from evaluation import evaluation_metrics

# Configuration variables
sc = var_scene_classification
oc = var_object_classification
r = var_RSV
f = file_paths 
L = Linemod_data('can')

[_, test] = L.getObjectSets()
[test_T, test_ply] = L.getTestFilesPaths()
em = evaluation_metrics(f.EVALUATION_METRICS_DATA)

initial_it = True
for it in range(len(test_ply)):
    
    if em.isSampleValid(test[it]):
        print('INFO: Sample ' + test[it] + ' already tested')
        continue    

    # Get the base name of the current sample 
    current_sample = test[it]   
    ''' -------------------------- Load point clouds -------------------------- '''
    start = time.time()
    print('INFO: Loading object point cloud')
    object = o3d.io.read_point_cloud(f.OBJECT)
    # Center the cloud in the origin
    object.translate(-object.get_center())
    print('INFO: Loading scene point cloud')
    scene = o3d.io.read_point_cloud(test_ply[it])
    scene = downsample(scene, v_size=5)
    end = time.time()
    print('\t TIME: ' + str(end - start))   

    ''' -------------------------- Data preparation -------------------------- '''
    start = time.time()
    T_gt = loadT(test_T[it])
    #object_T = applyGroundTruthPoseToObject(object, T_gt)
    #scene = cropScene(object_T, scene, radius = 500)
    print('INFO: Visualizing the object and scene cloud: ' + test_ply[it])   
    o3d.visualization.draw_geometries([object, scene])

    em.addData(current_sample, 'num_points_original', len(np.asarray(scene.points)))
    em.addData(current_sample, 'T_ground_truth', T_gt.tolist())
    end = time.time()
    print('\t TIME: ' + str(end - start))   
        
    ''' ------------------------- Scene segmentation -------------------------- '''
    start = time.time()
    print('INFO: Splitting the scene into segments for the classification module')
    scene_cs = cloud_splitter(scene, sc.INPUT_SHAPE)
    seeds = scene_cs.getSeeds(sc.VOXEL_SIZE_SCENE, sc.VOXEL_SIZE_SEEDS)
    segments_data = scene_cs.getSegments(seeds, sc.SEGMENT_RADIUS)
    # If no segments are found, stop the execution of the script.
    if segments_data is None:
        sys.exit()
    # Unpack the scene segments data
    [s_points, s_points_u, s_normals, s_indices, _] = segments_data  
    em.addData(current_sample, 'num_scene_segments', len(s_points))
    end = time.time()
    print('\t TIME: ' + str(end - start))   

    ''' ------------------------ Scene classification ------------------------- '''
    start = time.time()
    if initial_it is True:
        print('INFO: Running the scene classification PointNet module')
        scene_class = scene_classifier(sc.INPUT_SHAPE, sc.NUM_OF_CATEGORIES)
        print('INFO: Loading pre-trained PointNet model')
        scene_class.loadModel(f.SCN_CLASS_POINTNET_LOAD)

    print('INFO: Getting the predicted category for each segment')
    scores = scene_class.predict(s_points)
    # print('INFO: Visualizing the segments classified as object')
    scene_class.visualizePrediction(s_points_u, scores)
    print('INFO: From the segments classified as objects, select only the best candidate/s')
    c_indices = scene_class.getCandidates(scores, s_points, sc.NUM_OF_CANDIDATES)
    # Select the candidate data from the segments data
    c_points = s_points[c_indices]
    c_points_u = s_points_u[c_indices]
    c_normals = s_normals[c_indices]
    print('INFO: Visualizing the best candidate/s')
    scene_class.visualizeBestCandidates(c_points_u)
    # Reshape the candidate/s data in order to merge everything into one sample
    new_shape = (sc.NUM_OF_CANDIDATES*sc.NUM_OF_POINTS, sc.NUM_OF_DIMENSIONS)
    c_points = np.reshape(c_points, new_shape)
    c_points_u = np.reshape(c_points_u, new_shape)
    c_normals = np.reshape(c_normals, new_shape)
    # Since the candidates may share points among them, one we merged them, we need 
    # to remove the duplicates

    em.addData(current_sample, 'num_points_scene_candidates', len(c_points))
    
    print('INFO: Removing duplicate points')
    print('\tBefore: ' + str(len(c_points)))
    unq, unq_idx = np.unique(c_points_u, axis=0, return_index=True)
    c_points   = c_points[unq_idx]
    c_points_u   = c_points_u[unq_idx]
    c_normals  = c_normals[unq_idx] 
    print('\tAfter : ' + str(len(c_points)))

    em.addData(current_sample, 'num_points_scene_candidates_no_duplicates', len(c_points))    
    end = time.time()
    print('\t TIME: ' + str(end - start))   

    ''' ----------------------- Global feature vector ------------------------- '''
    start = time.time()
    # Convert candidate/s point data to point cloud
    candidate = fromNumpyArray(c_points_u, c_normals)
    print('INFO: Splitting the candidate into segments to compute the global feature vectors')
    g_cs = cloud_splitter(candidate, sc.INPUT_SHAPE)
    segments_data = g_cs.getSegments(candidate, sc.SEGMENT_RADIUS)
    # If no segments are found, stopt the execution of the script.
    if segments_data is None:
        sys.exit()
    # Unpack the object segments data
    [g_points, g_points_u, g_normals, g_seed_indices, _] = segments_data  
     
    em.addData(current_sample, 'num_gf_segments', len(g_points))
    
    c_features = scene_class.predictGlobalFeatureVectors(g_points)
    [n, _, _] = c_features.shape
    new_shape = (n, sc.NUM_OF_FEATURES)
    c_features = np.reshape(c_features, new_shape)

    print('INFO: Remove the points that didn\'t generate a valid segment')
    print('\tBefore: ' + str(len(c_points)))        
    c_points   = c_points[g_seed_indices]
    c_points_u   = c_points_u[g_seed_indices]
    c_normals  = c_normals[g_seed_indices] 
    print('\tAfter : ' + str(len(c_points)))

    em.addData(current_sample, 'num_points_scene_candidates_after_gf', len(c_points))
    end = time.time()
    print('\t TIME: ' + str(end - start))   


    ''' ------------------------- Object segmentation ------------------------- '''
    start = time.time()
    # Convert candidate/s point data to point cloud
    candidate = fromNumpyArray(c_points_u, c_normals)
    print('INFO: Splitting the object into segments for the classification module')
    object_cs = cloud_splitter(candidate, oc.INPUT_SHAPE)
    segments_data = object_cs.getSegments(candidate, oc.SEGMENT_RADIUS)
    # If no segments are found, stopt the execution of the script.
    if segments_data is None:
        sys.exit()
    # Unpack the object segments data
    [o_points, o_points_u, o_normals, o_seed_indices, _] = segments_data  
    print('INFO: Removing points that didn\'t generate a valid segment')
    print('\tBefore: ' + str(len(c_points)))
    c_points   = c_points[o_seed_indices]
    c_points_u = c_points_u[o_seed_indices]
    c_normals  = c_normals[o_seed_indices] 
    c_features = c_features[o_seed_indices] 
    print('\tAfter : ' + str(len(c_points)))

    em.addData(current_sample, 'num_points_scene_candidates_after_os', len(c_points))
    end = time.time()
    print('\t TIME: ' + str(end - start))   

    ''' ------------------------ Object classification ------------------------ '''
    start = time.time()
    if initial_it is True:
        print('INFO: Running the object classification PointNet module')
        object_class = object_classifier(oc.INPUT_SHAPE, oc.NUM_OF_CATEGORIES)
        print('INFO: Loading pre-trained PointNet model')
        object_class.loadModel(f.OBJ_CLASS_POINTNET_LOAD)        

    print('INFO: Getting the predicted category for each segment')
    scores = object_class.predict(o_points)
    predicted_categories = object_class.getPredictedCategory(scores, min_confidence=0.005)
    end = time.time()
    print('\t TIME: ' + str(end - start))   

    ''' ------------------- Rotation frame correspondences -------------------- '''
    start = time.time()
    print('INFO: Getting the correspondent rotation frame for each segment')
    R_seeds = loadSeedsRotationFrames(f.GTG_OBJECT_SEEDS_R)
    # Get the rotation frame associated to each sample
    predicted_R = np.asarray([R_seeds[p] for p in predicted_categories])

    good_points = np.array([], dtype=int)
    for i in range((len(predicted_categories)-1)):
        p = predicted_categories[i]
        if p is not -1:
            good_points = np.append(good_points, i)

    print('INFO: Filtering the points whose category wasn\'t confident enough')
    predicted_R = predicted_R[good_points]
    print('\tBefore: ' + str(len(c_points)))
    c_points   = c_points[good_points]
    c_points_u = c_points_u[good_points]
    c_normals  = c_normals[good_points] 
    c_features = c_features[good_points] 
    print('\tAfter: ' + str(len(c_points)))

    em.addData(current_sample, 'num_points_scene_candidates_after_correspondence', len(c_points))
    end = time.time()
    print('\t TIME: ' + str(end - start))   
        
    ''' ------------------------------- RSVnet -------------------------------- '''
    start = time.time()
    if initial_it is True:
        print('INFO: RSVnet model for predicting the scalar projection and radial vector length prediction')
        RSVnet = rsvnet_global_model(model_name='myRSVnet')
        print('INFO: Loading pre-trained RSVnet model')
        RSVnet.loadModel(f.RSVNET_PT_G)
        initial_it = False

    print('INFO: Getting the predicted values for each point')
    scores = RSVnet.predict(c_features)
    print('INFO: Casting the votes for each point of the cloud')
    v_data = RSVnet.getVotes_i(c_points_u, c_normals, scores, r.TESALLATION_LEVEL, vis=False)
    # Unpack the votes data to get the votes translation and rotation
    [votes, R_s, _] = v_data
    [m, n, _] = votes.shape
    print('\tVotes generated: ' + str(m*n))
    print('INFO: Getting the Rotation frame after applying the correspondence')
    temp = rsv(scene)
    R_r = temp.getRelativeRotationFrame(predicted_R, R_s)
    print('INFO: Clustering the votes')
    [density, t_candidates, R_candidates] = RSVnet.getDensity_i(c_points_u, c_normals, object, votes, R_r, r.SIGMA_t, r.SIGMA_R, vis=False)
    end = time.time()
    print('\t TIME: ' + str(end - start))   

    RSVnet.visualizeBestCandidates(scene, object, t_candidates, R_candidates)

    ''' -------------------------------- ICP ---------------------------------- '''
    start = time.time()
    T = buildT(t_candidates[0], R_candidates[0])

    em.addData(current_sample, 'T', T.tolist())

    icp_refinement  = icp(object, scene, T, threshold=20)

    T_icp_ptpoint = icp_refinement.point_to_point(vis=True)
    T_icp_ptplane = icp_refinement.point_to_plane(vis=True)

    em.addData(current_sample, 'T_icp_ptpoint', T_icp_ptpoint.tolist())
    em.addData(current_sample, 'T_icp_ptplane', T_icp_ptplane.tolist())

    em.saveMetrics(f.EVALUATION_METRICS_DATA)
    end = time.time()
    print('\t TIME: ' + str(end - start))   