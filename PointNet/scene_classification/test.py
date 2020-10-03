from configuration import file_paths, var_scene_classification
from PointNet_model import scene_classifier

# Configuration parameters
v = var_scene_classification
f = file_paths

# Declare the model
scene_class = scene_classifier(v.INPUT_SHAPE, v.NUM_OF_CATEGORIES)
# Load the pretrained model
scene_class.loadModel(f.SCN_CLASS_POINTNET_LOAD)
# Load .h5 files containing the data (points and labels)
[data_points, data_labels] = scene_class.loadDataset(f.SCN_CLASS_DATA)
#scene_class.visualizeData(num_of_samples=20)
# Prepare the data
data_splitted = scene_class.splitDataset(data_points, data_labels, v.TEST_SIZE)
# Unpack the splitted data
[_, _, test_points, test_labels] = data_splitted
scene_class.evaluate(test_points, test_labels)
