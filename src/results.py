import open3d as o3d

from configuration import Linemod_data, file_paths
from evaluation import evaluation_metrics
from pointcloud import downsample

f = file_paths
L = Linemod_data('can')

[_, test] = L.getObjectSets()
[_, test_ply] = L.getTestFilesPaths()
e = evaluation_metrics(f.EVALUATION_METRICS_DATA)

# Load object point cloud and perform preprocessing
object = o3d.io.read_point_cloud(f.OBJECT)
object = downsample(object, v_size=5)
diagonal = e.computeDiagonal(object, vis=False)

[P_0, P_1, P_2] = zip(*[e.evaluateOneSample(test[it], test_ply[it], object, method='RD', diagonal=diagonal, visualize=True) for it in range(len(test)) if e.isSampleValid(test[it])])

accuracy_0 = P_0.count('X')/len(P_0)*100
accuracy_1 = P_1.count('X')/len(P_1)*100
accuracy_2 = P_2.count('X')/len(P_2)*100

print(str(accuracy_0) + ' ,  ' +  str(accuracy_1) + ' ,  ' +  str(accuracy_2) )
