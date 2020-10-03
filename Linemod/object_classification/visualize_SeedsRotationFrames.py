import open3d as o3d

from configuration import file_paths, var_object_classification
from seeds_rotation_frames import (getSeedsRotationFrames,
                                   visualizeSeedsRotationFrames, saveSeedsRotationFrames)
from pointcloud import downsample
# Configuration parameters
f = file_paths
v = var_object_classification

object = o3d.io.read_point_cloud(f.OBJECT)
object.translate(-object.get_center())
# Downsample the point cloud with a voxel size big enough to reduce to 100 the
# number of points in the object model. This points will be used as seeds.
seeds = downsample(object, v.OBJECT_VOXEL_SIZE)

R_seeds = getSeedsRotationFrames(object, seeds)
visualizeSeedsRotationFrames(object, seeds, R_seeds)
saveSeedsRotationFrames(R_seeds, f.GTG_OBJECT_SEEDS_R)
