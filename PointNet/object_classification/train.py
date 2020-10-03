from configuration import file_paths, var_object_classification
from PointNet_model import object_classifier

# Configuration parameters
v = var_object_classification
f = file_paths

# Declare the model
object_class = object_classifier(v.INPUT_SHAPE, v.NUM_OF_CATEGORIES)
# Load .h5 files containing the data (points and labels)
[data_points, data_labels] = object_class.loadDataset(f.OBJ_CLASS_DATA_TRAIN)
#object_class.loadModel(f.OBJ_CLASS_POINTNET_LOAD)
# Prepare the data
# --> object_class.visualizeData(data_points, num_of_samples=20)
data_split = object_class.splitDataset(data_points, data_labels, 0.01)
# Unpack the split data
[X_train, Y_train, X_test, Y_test] = data_split
# Train the model
for i in range(v.NUM_OF_EPOCHS):
    print("Current epoch is: " + str(i))
    object_class.fit(X_train, Y_train, X_test, Y_test)
    # Save the trained model every fifth epoch
    if i % 1 == 0:
        object_class.saveModel(f.OBJ_CLASS_POINTNET_SAVE)
