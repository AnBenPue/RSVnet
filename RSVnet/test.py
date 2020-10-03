import os

import open3d as o3d

from configuration import file_paths, var_RSV
from RSVnet_model_global import rsvnet_global_model

# Configuration parameters
v = var_RSV
f = file_paths

# Load the path to the test files
files_testing = [f for f in os.listdir(f.RSVNET_DATA_TEST_G)] 
# Load a pretrained RSVnet 
RSV_net = rsvnet_global_model(model_name='myRSVnet')
# If a pretrained model exist, load it
if len(os.listdir(f.RSVNET_PT_G)) != 0:
    RSV_net.loadModel(f.RSVNET_PT_G) 
# Load model point cloud
object = o3d.io.read_point_cloud(f.OBJECT)
# Center the cloud in the origin
object.translate(-object.get_center())
       
for f_test in files_testing:
    print("INFO: Loading file: "  + f_test)
    file_path = os.path.join(f.RSVNET_DATA_TEST_G, f_test)
    [X_test, Y_test, points, normals, seeds, seeds_normals] = RSV_net.loadData(file_path)
    scores = RSV_net.predict(X_test)
    RSV_net.visualizeResults(points, normals, seeds, seeds_normals, scores, Y_test, num_of_points=5)

