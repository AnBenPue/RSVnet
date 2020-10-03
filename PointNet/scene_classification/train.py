from configuration import file_paths, var_scene_classification
from PointNet_model import scene_classifier

# Configuration parameters
v = var_scene_classification
f = file_paths

# Declare the model
scene_class = scene_classifier(v.INPUT_SHAPE, v.NUM_OF_CATEGORIES)
# Load .h5 files containing the data (points and labels)
[data_points, data_labels] = scene_class.loadDataset(f.SCN_CLASS_DATA)
# Prepare the data
# --> scene_class.visualizeData(data_points, num_of_samples=20)
data_splitted = scene_class.splitDataset(data_points, data_labels, v.TEST_SIZE)
# Unpack the splitted data
[X_train, Y_train, X_test, Y_test] = data_splitted
# Train the model
for i in range(v.NUM_OF_EPOCHS):
    scene_class.fit(X_train, Y_train, X_test, Y_test)
    # Save the trained model every fifth epoch
    if i % 5 == 0: 
        scene_class.saveModel(f.SCN_CLASS_POINTNET_SAVE)
