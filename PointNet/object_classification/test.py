from configuration import file_paths, var_object_classification
from PointNet_model import object_classifier

# Configuration parameters
v = var_object_classification
f = file_paths

# Declare the model
object_class = object_classifier(v.INPUT_SHAPE, v.NUM_OF_CATEGORIES)
# Load the pretrained model
object_class.loadModel(f.OBJ_CLASS_POINTNET_LOAD)
# Load .h5 files containing the data (points and labels)
[data_points, data_labels] = object_class.loadDataset(f.OBJ_CLASS_DATA_TEST)
# Prepare the data
data_splitted = object_class.splitDataset(data_points, data_labels, 0.99)
# Unpack the splitted data
[_, _, test_points, test_labels] = data_splitted
object_class.evaluate(test_points, test_labels)
